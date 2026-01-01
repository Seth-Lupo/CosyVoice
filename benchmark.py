#!/usr/bin/env python3
"""
CosyVoice2 Benchmark Script
Measures Time to First Audio (TTFA) for streaming inference
"""

import sys
import time
import statistics
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2

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

# English prompt audio and matching text
PROMPT_WAV = "./asset/cross_lingual_prompt.wav"


def run_benchmark(model_dir='pretrained_models/CosyVoice2-0.5B', num_warmup=2):
    """
    Run the full benchmark suite.
    """
    print("=" * 60)
    print("CosyVoice2 Benchmark - Time to First Audio (TTFA)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {model_dir}")
    load_start = time.perf_counter()
    model = CosyVoice2(model_dir=model_dir)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"Sample rate: {model.sample_rate} Hz")

    # Warmup runs
    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        warmup_phrase = "This is a warmup test. Please ignore this output."
        for _ in model.inference_cross_lingual(warmup_phrase, PROMPT_WAV, stream=True):
            pass
        print(f"  Warmup {i+1}/{num_warmup} complete")

    # Benchmark runs
    print(f"\nRunning benchmark on {len(TEST_PHRASES)} phrases...")
    print("-" * 60)

    ttfa_results = []

    for idx, phrase in enumerate(TEST_PHRASES):
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

    print("-" * 60)
    print("\nResults Summary:")
    print("=" * 60)
    print(f"Time to First Audio (TTFA):")
    print(f"  Mean:   {statistics.mean(ttfa_values):7.1f} ms")
    print(f"  Median: {statistics.median(ttfa_values):7.1f} ms")
    print(f"  Min:    {min(ttfa_values):7.1f} ms")
    print(f"  Max:    {max(ttfa_values):7.1f} ms")
    print(f"  Stdev:  {statistics.stdev(ttfa_values):7.1f} ms")

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

    print("=" * 60)

    return ttfa_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CosyVoice2 TTFA Benchmark')
    parser.add_argument('--model_dir', type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='Path to the model directory')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup iterations')

    args = parser.parse_args()

    results = run_benchmark(model_dir=args.model_dir, num_warmup=args.warmup)
