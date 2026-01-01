#!/bin/bash
# CosyVoice2 TensorRT-LLM Inference Server
# Launches high-performance TTS server with TensorRT acceleration

set -e

# Configuration
MODEL_DIR="${MODEL_DIR:-pretrained_models/CosyVoice2-0.5B}"
TRT_ENGINES_DIR="${TRT_ENGINES_DIR:-trt_engines}"
SERVER_PORT="${SERVER_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"

echo "============================================================"
echo "CosyVoice2 TensorRT-LLM Server"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model Dir:     $MODEL_DIR"
echo "  TRT Engines:   $TRT_ENGINES_DIR"
echo "  HTTP Port:     $SERVER_PORT"
echo "  gRPC Port:     $GRPC_PORT"
echo "  Metrics Port:  $METRICS_PORT"
echo ""

# Check if TensorRT engines exist
if [ ! -d "$TRT_ENGINES_DIR" ]; then
    echo "WARNING: TensorRT engines not found at $TRT_ENGINES_DIR"
    echo "Run ./build_trt_engine.sh first to build the engines."
    echo ""
    echo "Continuing with PyTorch backend..."
    USE_TRT=false
else
    echo "TensorRT engines found!"
    USE_TRT=true
fi

# Check for TensorRT flow decoder
FLOW_TRT="$MODEL_DIR/flow.decoder.estimator.fp16.plan"
if [ -f "$FLOW_TRT" ]; then
    echo "TensorRT flow decoder found: $FLOW_TRT"
    LOAD_TRT_FLAG="--load_trt"
else
    echo "TensorRT flow decoder not found, using PyTorch"
    LOAD_TRT_FLAG=""
fi

echo ""
echo "============================================================"
echo "Starting Server..."
echo "============================================================"
echo ""

# Option 1: Run with Gradio WebUI (simple)
run_webui() {
    echo "Starting Gradio WebUI server..."
    python3 webui.py \
        --port $SERVER_PORT \
        --model_dir $MODEL_DIR
}

# Option 2: Run with FastAPI (production)
run_fastapi() {
    echo "Starting FastAPI server..."
    python3 -c "
import sys
sys.path.append('third_party/Matcha-TTS')

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model('CosyVoice2ForCausalLM', CosyVoice2ForCausalLM)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import CosyVoice2
import io
import torch
import torchaudio
import uvicorn

app = FastAPI(title='CosyVoice2 TTS API')

# Load model with TensorRT
print('Loading CosyVoice2 with TensorRT acceleration...')
model = CosyVoice2(
    model_dir='$MODEL_DIR',
    load_jit=True,
    load_trt=$([[ \"$USE_TRT\" == \"true\" ]] && echo \"True\" || echo \"False\"),
    load_vllm=True,
    fp16=True
)
print(f'Model loaded! Sample rate: {model.sample_rate}')


@app.get('/health')
async def health():
    return {'status': 'healthy', 'sample_rate': model.sample_rate}


@app.post('/tts/stream')
async def tts_stream(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    '''Stream TTS audio as it's generated'''
    # Save uploaded prompt audio
    prompt_path = '/tmp/prompt.wav'
    with open(prompt_path, 'wb') as f:
        f.write(await prompt_audio.read())

    async def generate():
        for output in model.inference_cross_lingual(text, prompt_path, stream=True):
            audio = output['tts_speech']
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio, model.sample_rate, format='wav')
            buffer.seek(0)
            yield buffer.read()

    return StreamingResponse(generate(), media_type='audio/wav')


@app.post('/tts')
async def tts(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    '''Generate complete TTS audio'''
    # Save uploaded prompt audio
    prompt_path = '/tmp/prompt.wav'
    with open(prompt_path, 'wb') as f:
        f.write(await prompt_audio.read())

    # Generate all audio
    audio_chunks = []
    for output in model.inference_cross_lingual(text, prompt_path, stream=False):
        audio_chunks.append(output['tts_speech'])

    # Concatenate all chunks
    full_audio = torch.cat(audio_chunks, dim=1)

    # Return as WAV
    buffer = io.BytesIO()
    torchaudio.save(buffer, full_audio, model.sample_rate, format='wav')
    buffer.seek(0)

    return StreamingResponse(buffer, media_type='audio/wav')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=$SERVER_PORT)
"
}

# Option 3: Run with Triton Inference Server (enterprise)
run_triton() {
    echo "Starting Triton Inference Server..."

    # Check if tritonserver is available
    if ! command -v tritonserver &> /dev/null; then
        echo "ERROR: tritonserver not found"
        echo "Install with: pip install tritonclient[all]"
        exit 1
    fi

    # Prepare Triton model repository
    TRITON_REPO="triton_model_repository"
    mkdir -p $TRITON_REPO/cosyvoice2/1

    # Create Triton config
    cat > $TRITON_REPO/cosyvoice2/config.pbtxt << EOF
name: "cosyvoice2"
backend: "python"
max_batch_size: 16

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "prompt_audio"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

output [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
EOF

    tritonserver \
        --model-repository=$TRITON_REPO \
        --http-port=$SERVER_PORT \
        --grpc-port=$GRPC_PORT \
        --metrics-port=$METRICS_PORT
}

# Parse command line
case "${1:-webui}" in
    webui)
        run_webui
        ;;
    fastapi)
        run_fastapi
        ;;
    triton)
        run_triton
        ;;
    *)
        echo "Usage: $0 {webui|fastapi|triton}"
        echo ""
        echo "  webui   - Gradio web interface (default)"
        echo "  fastapi - FastAPI REST server"
        echo "  triton  - Triton Inference Server"
        exit 1
        ;;
esac
