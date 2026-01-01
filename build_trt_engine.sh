#!/bin/bash
# CosyVoice2 TensorRT-LLM Engine Build Script
# Builds optimized FP16 engines for low-latency inference

set -e

# Configuration
MODEL_DIR="${MODEL_DIR:-pretrained_models/CosyVoice2-0.5B}"
TRT_WEIGHTS_DIR="${TRT_WEIGHTS_DIR:-trt_weights}"
TRT_ENGINES_DIR="${TRT_ENGINES_DIR:-trt_engines}"
DTYPE="${DTYPE:-float16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-32768}"
TP_SIZE="${TP_SIZE:-1}"

echo "============================================================"
echo "CosyVoice2 TensorRT-LLM Engine Builder"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model Dir:       $MODEL_DIR"
echo "  TRT Weights:     $TRT_WEIGHTS_DIR"
echo "  TRT Engines:     $TRT_ENGINES_DIR"
echo "  Data Type:       $DTYPE"
echo "  Max Batch Size:  $MAX_BATCH_SIZE"
echo "  Max Tokens:      $MAX_NUM_TOKENS"
echo "  Tensor Parallel: $TP_SIZE"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Please download the model first:"
    echo "  python -c \"from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$MODEL_DIR')\""
    exit 1
fi

# Check for CosyVoice-BlankEN (Qwen weights)
QWEN_DIR="$MODEL_DIR/CosyVoice-BlankEN"
if [ ! -d "$QWEN_DIR" ]; then
    echo "ERROR: Qwen model not found: $QWEN_DIR"
    exit 1
fi

echo "Step 1: Converting Qwen checkpoint to TensorRT-LLM format..."
echo "------------------------------------------------------------"

mkdir -p $TRT_WEIGHTS_DIR

# Check if trtllm-build is available
if ! command -v trtllm-build &> /dev/null; then
    echo "WARNING: trtllm-build not found in PATH"
    echo "Trying to use convert_checkpoint.py directly..."

    # Use the conversion script from the repo
    python3 runtime/triton_trtllm/scripts/convert_checkpoint.py \
        --model_dir $QWEN_DIR \
        --output_dir $TRT_WEIGHTS_DIR \
        --dtype $DTYPE \
        --tp_size $TP_SIZE
else
    # Use trtllm-build's convert checkpoint
    python3 -c "
from tensorrt_llm.models import QWenForCausalLM
from tensorrt_llm import Mapping

print('Loading Qwen model from: $QWEN_DIR')
mapping = Mapping(world_size=$TP_SIZE, tp_size=$TP_SIZE)
qwen = QWenForCausalLM.from_hugging_face(
    '$QWEN_DIR',
    dtype='$DTYPE',
    mapping=mapping
)
print('Saving TRT-LLM checkpoint to: $TRT_WEIGHTS_DIR')
qwen.save_checkpoint('$TRT_WEIGHTS_DIR')
print('Checkpoint conversion complete!')
"
fi

echo ""
echo "Step 2: Building TensorRT-LLM engine..."
echo "------------------------------------------------------------"

mkdir -p $TRT_ENGINES_DIR

# Build the engine
trtllm-build \
    --checkpoint_dir $TRT_WEIGHTS_DIR \
    --output_dir $TRT_ENGINES_DIR \
    --max_batch_size $MAX_BATCH_SIZE \
    --max_num_tokens $MAX_NUM_TOKENS \
    --gemm_plugin $DTYPE \
    --gpt_attention_plugin $DTYPE \
    --remove_input_padding enable \
    --paged_kv_cache enable \
    --use_fused_mlp enable \
    --context_fmha enable

echo ""
echo "Step 3: Exporting Flow Decoder to TensorRT..."
echo "------------------------------------------------------------"

# Export the flow decoder estimator to ONNX and TRT
python3 -c "
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
import torch
import os

print('Loading CosyVoice2 model...')
model = CosyVoice2(model_dir='$MODEL_DIR', fp16=True)

# Export flow decoder estimator to ONNX
estimator = model.model.flow.decoder.estimator
estimator.eval()

device = model.model.device
batch_size, seq_len = 2, 256
out_channels = estimator.out_channels

x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
t = torch.rand((batch_size), dtype=torch.float32, device=device)
spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)

onnx_path = '$MODEL_DIR/flow.decoder.estimator.fp16.onnx'
print(f'Exporting to ONNX: {onnx_path}')

torch.onnx.export(
    estimator.half(),
    (x.half(), mask.half(), mu.half(), t.half(), spks.half(), cond.half()),
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
    output_names=['estimator_out'],
    dynamic_axes={
        'x': {0: 'batch', 2: 'seq_len'},
        'mask': {0: 'batch', 2: 'seq_len'},
        'mu': {0: 'batch', 2: 'seq_len'},
        'cond': {0: 'batch', 2: 'seq_len'},
        'estimator_out': {0: 'batch', 2: 'seq_len'},
    }
)
print('ONNX export complete!')
"

# Convert ONNX to TensorRT
ONNX_PATH="$MODEL_DIR/flow.decoder.estimator.fp16.onnx"
TRT_PATH="$MODEL_DIR/flow.decoder.estimator.fp16.plan"

if [ -f "$ONNX_PATH" ]; then
    echo "Converting ONNX to TensorRT engine..."
    /usr/local/tensorrt/bin/trtexec \
        --onnx=$ONNX_PATH \
        --saveEngine=$TRT_PATH \
        --fp16 \
        --minShapes=x:1x80x4,mask:1x1x4,mu:1x80x4,t:1,spks:1x80,cond:1x80x4 \
        --optShapes=x:2x80x500,mask:2x1x500,mu:2x80x500,t:2,spks:2x80,cond:2x80x500 \
        --maxShapes=x:16x80x3000,mask:16x1x3000,mu:16x80x3000,t:16,spks:16x80,cond:16x80x3000 \
        --workspace=4096
    echo "TensorRT engine saved to: $TRT_PATH"
else
    echo "WARNING: ONNX file not found, skipping TensorRT conversion"
fi

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Engine locations:"
echo "  LLM Engine:  $TRT_ENGINES_DIR/"
echo "  Flow Engine: $TRT_PATH"
echo ""
echo "To run the TensorRT benchmark:"
echo "  python benchmark_trt.py"
echo ""
