"""
Streaming voice-clone inference without torch.compile.

Usage:
    python examples/test_streaming.py
"""

import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def main():
    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    start = log_time(start, "Load Base model")

    ref_audio_path = "bogd-sample.wav"
    ref_text = (
        "Ого! Ёж. В моём мире нет таких зверей, могу я с тобой подружиться? Кроме тебя, со мной никто "
        "не разговаривает, я не хочу быть одна. Этот лес такой большой! Деревья так высоки! "
        "И их кроны затмевают солнечный свет! Спасибо тебе, мой новый друг. За яблочко и экскурсию "
        "по вашему потрясающему миру, я буду ждать нашей новой встречи. Пока!"
    )

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    start = log_time(start, "Create voice clone prompt")

    test_text = "Hello! This is a streaming voice clone test."

    print("\n--- Standard generation ---")
    t0 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=test_text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    print(f"[{time.time() - t0:.2f}s] Standard generate")
    sf.write("clone_standard.wav", wavs[0], sr)

    print("\n--- Streaming generation ---")
    t0 = time.time()
    chunks = []
    first_chunk_time = None
    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=test_text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
        decode_window_frames=80,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - t0
            print(f"[{first_chunk_time:.2f}s] First chunk received ({len(chunk)} samples)")

    print(f"[{time.time() - t0:.2f}s] Streaming complete ({len(chunks)} chunks)")
    sf.write("clone_streaming.wav", np.concatenate(chunks), chunk_sr)


if __name__ == "__main__":
    main()
