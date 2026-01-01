#!/usr/bin/env python3
"""
Export CosyVoice2 Flow Decoder to TensorRT
Standalone script that avoids torchaudio import issues
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add Matcha-TTS to path
sys.path.insert(0, 'third_party/Matcha-TTS')


def load_flow_decoder(model_dir: str, device: str = 'cuda'):
    """Load just the flow decoder without full CosyVoice initialization"""
    from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
    from omegaconf import OmegaConf

    # Load config (try both naming conventions)
    config_path = os.path.join(model_dir, 'cosyvoice2.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(model_dir, 'cosyvoice.yaml')

    with open(config_path, 'r') as f:
        import yaml
        configs = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    # Get flow config
    flow_config = configs.get('flow', {})

    # Build flow model
    flow = CausalMaskedDiffWithXvec(
        in_channels=flow_config.get('in_channels', 80),
        out_channel=flow_config.get('out_channel', 80),
        spk_emb_dim=flow_config.get('spk_emb_dim', 192),
        n_spks=flow_config.get('n_spks', 1),
        cfm_params=OmegaConf.create(flow_config.get('cfm_params', {})),
        decoder_params=OmegaConf.create(flow_config.get('decoder', {}).get('params', {})),
    )

    # Load weights
    flow_ckpt = os.path.join(model_dir, 'flow.pt')
    if os.path.exists(flow_ckpt):
        state_dict = torch.load(flow_ckpt, map_location='cpu', weights_only=True)
        flow.load_state_dict(state_dict)
        print(f"Loaded flow weights from {flow_ckpt}")
    else:
        raise FileNotFoundError(f"Flow checkpoint not found: {flow_ckpt}")

    flow = flow.to(device)
    flow.eval()

    return flow


def export_estimator_onnx(flow, output_path: str, device: str = 'cuda'):
    """Export the flow decoder estimator to ONNX"""
    estimator = flow.decoder.estimator
    estimator.eval()

    # Get dimensions
    out_channels = estimator.out_channels
    batch_size, seq_len = 2, 256

    # Create dummy inputs
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float16, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)
    t = torch.rand((batch_size,), dtype=torch.float16, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float16, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float16, device=device)

    # Convert estimator to FP16
    estimator_fp16 = estimator.half()

    print(f"Exporting estimator to ONNX: {output_path}")
    print(f"  Input shapes: x={x.shape}, mask={mask.shape}, mu={mu.shape}, t={t.shape}, spks={spks.shape}, cond={cond.shape}")

    torch.onnx.export(
        estimator_fp16,
        (x, mask, mu, t, spks, cond),
        output_path,
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
    print("ONNX export complete!")
    return output_path


def convert_onnx_to_trt(onnx_path: str, trt_path: str, trtexec_path: str = '/usr/local/tensorrt/bin/trtexec'):
    """Convert ONNX to TensorRT using trtexec"""
    import subprocess

    cmd = [
        trtexec_path,
        f'--onnx={onnx_path}',
        f'--saveEngine={trt_path}',
        '--fp16',
        '--minShapes=x:1x80x4,mask:1x1x4,mu:1x80x4,t:1,spks:1x80,cond:1x80x4',
        '--optShapes=x:2x80x500,mask:2x1x500,mu:2x80x500,t:2,spks:2x80,cond:2x80x500',
        '--maxShapes=x:16x80x3000,mask:16x1x3000,mu:16x80x3000,t:16,spks:16x80,cond:16x80x3000',
        '--workspace=4096',
    ]

    print(f"Converting ONNX to TensorRT...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"TensorRT engine saved to: {trt_path}")
    else:
        print(f"TensorRT conversion failed with code {result.returncode}")
        return None

    return trt_path


def main():
    parser = argparse.ArgumentParser(description='Export CosyVoice2 Flow Decoder to TensorRT')
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B',
                        help='Path to CosyVoice2 model directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to model_dir)')
    parser.add_argument('--trtexec', type=str, default='/usr/local/tensorrt/bin/trtexec',
                        help='Path to trtexec binary')
    parser.add_argument('--onnx-only', action='store_true',
                        help='Only export ONNX, skip TensorRT conversion')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for export')

    args = parser.parse_args()

    output_dir = args.output_dir or args.model_dir
    onnx_path = os.path.join(output_dir, 'flow.decoder.estimator.fp16.onnx')
    trt_path = os.path.join(output_dir, 'flow.decoder.estimator.fp16.plan')

    print("=" * 60)
    print("CosyVoice2 Flow Decoder TensorRT Export")
    print("=" * 60)
    print(f"Model Dir:  {args.model_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Device:     {args.device}")
    print("")

    # Load flow decoder
    print("Loading flow decoder...")
    flow = load_flow_decoder(args.model_dir, args.device)

    # Export to ONNX
    export_estimator_onnx(flow, onnx_path, args.device)

    # Convert to TensorRT
    if not args.onnx_only:
        if os.path.exists(args.trtexec):
            convert_onnx_to_trt(onnx_path, trt_path, args.trtexec)
        else:
            print(f"WARNING: trtexec not found at {args.trtexec}")
            print("Run manually:")
            print(f"  {args.trtexec} --onnx={onnx_path} --saveEngine={trt_path} --fp16 ...")

    print("")
    print("=" * 60)
    print("Export Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
