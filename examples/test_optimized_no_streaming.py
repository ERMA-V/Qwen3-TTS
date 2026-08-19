"""
Non-streaming generation with the same talker/decoder compile optimizations.

Usage:
    python examples/test_optimized_no_streaming.py
"""

import time

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")


def main():
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
    test_text = "Hello! This is a non-streaming optimized generation test."

    print("Baseline generate...")
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=test_text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    print(f"Baseline: {time.time() - t0:.2f}s")
    sf.write("output_baseline.wav", wavs[0], sr)

    print("Enabling optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=300,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="max-autotune",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    print("Warmup...")
    model.generate_voice_clone(
        text="Warmup one two three.",
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )

    print("Optimized generate...")
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=test_text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    print(f"Optimized: {time.time() - t0:.2f}s")
    sf.write("output_optimized.wav", wavs[0], sr)


if __name__ == "__main__":
    main()
