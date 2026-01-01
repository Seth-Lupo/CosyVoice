#!/usr/bin/env python3
"""
CosyVoice2 TensorRT-LLM Benchmark Script
Measures Time to First Audio (TTFA) with TensorRT acceleration
"""

import sys
import time
import statistics
sys.path.append('third_party/Matcha-TTS')

# Register vLLM model before imports
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.common import set_all_random_seed

# 10 two-sentence test phrases (English)
TEST_PHRASES = [
    "The sun rose over the mountains. Birds began to sing in the trees.",
    "She opened the door slowly. A cold breeze swept through the hallway.",
    "The coffee was still hot. He took a careful sip and smiled.",
    "Rain began to fall outside. The streets were soon covered in puddles.",
    "The meeting started at noon. Everyone gathered around the conference table.",
    "Music played softly in the background. Couples danced under the starlight.",
    "The train arrived on time. Passengers hurried to find their seats.",
    "Autumn leaves covered the path. Children ran through them laughing.",
    "The book was finally finished. She closed it with a sense of accomplishment.",
    "Waves crashed against the shore. Seagulls circled overhead in the evening sky.",
]

# English prompt audio
PROMPT_WAV = "./asset/cross_lingual_prompt.wav"


def run_trt_benchmark(model_dir='pretrained_models/CosyVoice2-0.5B', num_warmup=3,
                      load_jit=True, load_trt=True, load_vllm=True):
    """
    Run TensorRT-accelerated benchmark suite.
    """
    print("=" * 70)
    print("CosyVoice2 TensorRT-LLM Benchmark - Time to First Audio (TTFA)")
    print("=" * 70)

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Model Dir:   {model_dir}")
    print(f"  Precision:   FP16")
    print(f"  JIT:         {'Enabled' if load_jit else 'Disabled'}")
    print(f"  TensorRT:    {'Enabled' if load_trt else 'Disabled'}")
    print(f"  vLLM:        {'Enabled' if load_vllm else 'Disabled'}")

    # Load model with TensorRT
    print(f"\nLoading model with TensorRT acceleration...")
    load_start = time.perf_counter()
    model = CosyVoice2(
        model_dir=model_dir,
        load_jit=load_jit,
        load_trt=load_trt,
        load_vllm=load_vllm,
        fp16=True
    )
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Sample rate: {model.sample_rate} Hz")

    # Warmup runs (more warmups for TRT to optimize)
    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        set_all_random_seed(i)
        warmup_phrase = "This is a warmup test. Please ignore this output."
        for _ in model.inference_cross_lingual(warmup_phrase, PROMPT_WAV, stream=True):
            pass
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark runs
    print(f"\nRunning benchmark on {len(TEST_PHRASES)} phrases...")
    print("-" * 70)

    ttfa_results = []

    for idx, phrase in enumerate(TEST_PHRASES):
        set_all_random_seed(42 + idx)  # Consistent seeds for reproducibility

        # Measure TTFA with streaming
        start_time = time.perf_counter()
        first_chunk_time = None
        total_audio_duration = 0
        chunk_count = 0

        for output in model.inference_cross_lingual(
            phrase,
            PROMPT_WAV,
            stream=True
        ):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start_time) * 1000
            chunk_count += 1
            total_audio_duration += output['tts_speech'].shape[1] / model.sample_rate

        total_time = (time.perf_counter() - start_time) * 1000

        ttfa_results.append({
            'trial': idx + 1,
            'ttfa_ms': first_chunk_time,
            'total_time_ms': total_time,
            'audio_duration_s': total_audio_duration,
            'chunks': chunk_count,
            'phrase_len': len(phrase)
        })

        print(f"Trial {idx+1:2d}: TTFA = {first_chunk_time:7.1f} ms | "
              f"Total = {total_time:7.1f} ms | "
              f"Audio = {total_audio_duration:.2f}s | "
              f"Chunks = {chunk_count}")

    # Calculate statistics
    ttfa_values = [r['ttfa_ms'] for r in ttfa_results]
    total_values = [r['total_time_ms'] for r in ttfa_results]

    print("-" * 70)
    print("\nResults Summary (TensorRT-LLM FP16):")
    print("=" * 70)
    print(f"Time to First Audio (TTFA):")
    print(f"  Mean:   {statistics.mean(ttfa_values):7.1f} ms")
    print(f"  Median: {statistics.median(ttfa_values):7.1f} ms")
    print(f"  Min:    {min(ttfa_values):7.1f} ms")
    print(f"  Max:    {max(ttfa_values):7.1f} ms")
    print(f"  Stdev:  {statistics.stdev(ttfa_values):7.1f} ms")
    print(f"  P95:    {sorted(ttfa_values)[int(len(ttfa_values)*0.95)]:7.1f} ms")

    print(f"\nTotal Generation Time:")
    print(f"  Mean:   {statistics.mean(total_values):7.1f} ms")
    print(f"  Median: {statistics.median(total_values):7.1f} ms")
    print(f"  Min:    {min(total_values):7.1f} ms")
    print(f"  Max:    {max(total_values):7.1f} ms")

    # Calculate RTF (Real-Time Factor)
    rtf_values = [r['total_time_ms'] / (r['audio_duration_s'] * 1000) for r in ttfa_results]
    print(f"\nReal-Time Factor (RTF):")
    print(f"  Mean:   {statistics.mean(rtf_values):.3f}x")
    print(f"  Best:   {min(rtf_values):.3f}x")

    # Throughput
    total_audio = sum(r['audio_duration_s'] for r in ttfa_results)
    total_gen_time = sum(r['total_time_ms'] for r in ttfa_results) / 1000
    print(f"\nThroughput:")
    print(f"  Audio generated: {total_audio:.2f}s")
    print(f"  Wall time:       {total_gen_time:.2f}s")
    print(f"  Throughput:      {total_audio/total_gen_time:.2f}x realtime")

    print("=" * 70)

    return ttfa_results


def compare_with_baseline(model_dir='pretrained_models/CosyVoice2-0.5B'):
    """
    Run comparison between TensorRT and non-TensorRT inference.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: TensorRT vs Baseline")
    print("=" * 70 + "\n")

    # Run baseline (FP16 without TRT)
    print(">>> Running Baseline (FP16, no TensorRT)...")
    print("-" * 70)

    baseline_model = CosyVoice2(model_dir=model_dir, fp16=True)

    baseline_ttfa = []
    for idx, phrase in enumerate(TEST_PHRASES[:5]):  # Only 5 for quick comparison
        set_all_random_seed(42 + idx)
        start_time = time.perf_counter()
        first_chunk_time = None

        for output in baseline_model.inference_cross_lingual(phrase, PROMPT_WAV, stream=True):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start_time) * 1000
                break

        baseline_ttfa.append(first_chunk_time)
        print(f"  Baseline Trial {idx+1}: TTFA = {first_chunk_time:.1f} ms")

    del baseline_model

    # Run TensorRT
    print("\n>>> Running TensorRT-LLM (FP16)...")
    print("-" * 70)

    trt_model = CosyVoice2(
        model_dir=model_dir,
        load_jit=True,
        load_trt=True,
        load_vllm=True,
        fp16=True
    )

    # Extra warmup for TRT
    for i in range(3):
        for _ in trt_model.inference_cross_lingual("Warmup phrase.", PROMPT_WAV, stream=True):
            pass

    trt_ttfa = []
    for idx, phrase in enumerate(TEST_PHRASES[:5]):
        set_all_random_seed(42 + idx)
        start_time = time.perf_counter()
        first_chunk_time = None

        for output in trt_model.inference_cross_lingual(phrase, PROMPT_WAV, stream=True):
            if first_chunk_time is None:
                first_chunk_time = (time.perf_counter() - start_time) * 1000
                break

        trt_ttfa.append(first_chunk_time)
        print(f"  TensorRT Trial {idx+1}: TTFA = {first_chunk_time:.1f} ms")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    baseline_mean = statistics.mean(baseline_ttfa)
    trt_mean = statistics.mean(trt_ttfa)
    speedup = baseline_mean / trt_mean

    print(f"  Baseline Mean TTFA:  {baseline_mean:7.1f} ms")
    print(f"  TensorRT Mean TTFA:  {trt_mean:7.1f} ms")
    print(f"  Speedup:             {speedup:7.2f}x")
    print(f"  Latency Reduction:   {(1 - trt_mean/baseline_mean)*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CosyVoice2 TensorRT-LLM TTFA Benchmark')
    parser.add_argument('--model_dir', type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='Path to the model directory')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--no-jit', action='store_true',
                        help='Disable JIT compilation')
    parser.add_argument('--no-trt', action='store_true',
                        help='Disable TensorRT for flow decoder')
    parser.add_argument('--no-vllm', action='store_true',
                        help='Disable vLLM for LLM inference')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison with baseline (no TRT)')

    args = parser.parse_args()

    if args.compare:
        compare_with_baseline(model_dir=args.model_dir)
    else:
        results = run_trt_benchmark(
            model_dir=args.model_dir,
            num_warmup=args.warmup,
            load_jit=not args.no_jit,
            load_trt=not args.no_trt,
            load_vllm=not args.no_vllm
        )
