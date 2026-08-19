"""
Stream PCM from a CustomVoice / finetuned 12Hz checkpoint.

Finetuned Base checkpoints are exported as tts_model_type="custom_voice"
and use stream_generate_custom_voice(), not stream_generate_voice_clone().

Usage:
    python examples/test_streaming_custom_voice.py
"""

import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision("high")

MODEL_PATH = "output/checkpoint-epoch-2"
SPEAKER = "erma-v"
LANGUAGE = "English"
EMIT_EVERY = 4
DECODE_WINDOW = 80


def stream_once(model, text: str, label: str):
    start = time.time()
    chunks = []
    first_chunk_time = None
    sample_rate = 24000

    for chunk, sample_rate in model.stream_generate_custom_voice(
        text=text,
        speaker=SPEAKER,
        language=LANGUAGE,
        emit_every_frames=EMIT_EVERY,
        decode_window_frames=DECODE_WINDOW,
    ):
        chunks.append(chunk)
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"[{label}] first chunk: {first_chunk_time:.2f}s ({len(chunk)} samples)")

    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    total = time.time() - start
    duration = len(audio) / sample_rate if sample_rate else 0
    rtf = total / duration if duration else 0
    print(
        f"[{label}] total={total:.2f}s audio={duration:.2f}s "
        f"chunks={len(chunks)} RTF={rtf:.2f}"
    )
    return audio, sample_rate


def main():
    print("Loading finetuned custom-voice model...")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    test_text = "She said she would be here by noon."

    print("\n--- Streaming without compile ---")
    audio, sr = stream_once(model, test_text, "baseline")
    sf.write("output_custom_streaming_baseline.wav", audio, sr)

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

    print("\nWarmup (compilation happens here)...")
    stream_once(model, "Warmup one two three.", "warmup")

    print("\n--- Streaming with compile ---")
    audio, sr = stream_once(model, test_text, "optimized")
    sf.write("output_custom_streaming_optimized.wav", audio, sr)


if __name__ == "__main__":
    main()
