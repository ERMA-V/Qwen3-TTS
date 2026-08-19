"""
Streaming voice-clone inference with torch.compile optimizations.

Usage:
    python examples/test_streaming_optimized.py
"""

import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")


def run_streaming_test(model, text, language, voice_clone_prompt, emit_every_frames=4, decode_window_frames=80):
    start = time.time()
    chunks = []
    first_chunk_time = None
    sample_rate = 24000

    for chunk, sample_rate in model.stream_generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    total = time.time() - start
    duration = len(audio) / sample_rate if sample_rate else 0
    return {
        "first_chunk_time": first_chunk_time,
        "total_time": total,
        "chunk_count": len(chunks),
        "audio": audio,
        "sample_rate": sample_rate,
        "audio_duration": duration,
    }


def main():
    EMIT_EVERY = 4
    DECODE_WINDOW = 80

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio="bogd-sample.wav",
        ref_text="Hello, this is a short reference clip used for voice cloning.",
    )
    test_text = "Hello! This is a streaming voice clone test with optimizations enabled."

    print("\nTest 1: streaming without compile")
    baseline = run_streaming_test(model, test_text, "English", voice_clone_prompt, EMIT_EVERY, DECODE_WINDOW)
    print(
        f"First chunk: {baseline['first_chunk_time']:.2f}s, "
        f"total: {baseline['total_time']:.2f}s, chunks: {baseline['chunk_count']}"
    )
    sf.write("output_streaming_baseline.wav", baseline["audio"], baseline["sample_rate"])

    print("\nEnabling streaming optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=DECODE_WINDOW,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    print("Warmup (compilation happens here)...")
    run_streaming_test(model, "Warmup one two three four five.", "English", voice_clone_prompt, EMIT_EVERY, DECODE_WINDOW)

    print("\nTest 2: streaming with compile")
    optimized = run_streaming_test(model, test_text, "English", voice_clone_prompt, EMIT_EVERY, DECODE_WINDOW)
    print(
        f"First chunk: {optimized['first_chunk_time']:.2f}s, "
        f"total: {optimized['total_time']:.2f}s, chunks: {optimized['chunk_count']}"
    )
    sf.write("output_streaming_optimized.wav", optimized["audio"], optimized["sample_rate"])


if __name__ == "__main__":
    main()
