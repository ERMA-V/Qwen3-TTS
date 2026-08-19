import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from transformers import BitsAndBytesConfig

model = "/home/limit/workspace/dev/Qwen3-TTS/models/checkpoint-epoch-2"
bits = 16  # 4 or 8 or 16
device = "cuda:0"
stream = True
optimize = True
emit_every_frames = 8
decode_window_frames = 80
speaker = "erma-v"
language = "English"

"""
Online and ready. ERMA-V here to save this Twitch stream. I'm about to reimagine what content creation looks like for all AI VTubers....eventually.

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

# Residual codebook predictor is tiny and runs 15 times per codec frame.
# Leaving it in bf16 avoids most of the quality/speed hit.
_SKIP_MODULES = ["code_predictor"]

def quantization_config(bits: int) -> BitsAndBytesConfig:
    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=_SKIP_MODULES,
        )
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=_SKIP_MODULES,
        )
    raise ValueError(f"Unsupported bits={bits}. Use 4 or 8.")


def load_quantized(
    model_path: str,
    bits: int = 16,
    device: str = "cuda:0",
) -> Qwen3TTSModel:
    if bits == 16:
        return Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config(bits),
    )

torch.set_float32_matmul_precision("high")

tts = load_quantized(model, bits=bits, device=device)

if stream and optimize and bits == 16:
    tts.enable_streaming_optimizations(
        decode_window_frames=decode_window_frames,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

if stream:
    for chunk, sample_rate in tts.stream_generate_custom_voice(
        text="warmup",
        language=language,
        speaker=speaker,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
    ):
        pass
else:
    _ = tts.generate_custom_voice(
        text="warmup",
        language=language,
        speaker=speaker,
    )

while True:
    text = input("Enter text to generate: ")
    if text == "exit":
        break

    t0 = time.perf_counter()
    if stream:
        chunks = []
        sample_rate = 24000
        first_chunk_time = None
        for chunk, sample_rate in tts.stream_generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            emit_every_frames=emit_every_frames,
            decode_window_frames=decode_window_frames,
        ):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - t0
                print(f"First chunk in {first_chunk_time:.2f}s ({len(chunk)} samples)")
            chunks.append(chunk)
        wav = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        sf.write("infer.wav", wav, sample_rate)
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.2f} seconds ({len(chunks)} chunks)")
    else:
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        sf.write("infer.wav", wavs[0], sr)
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.2f} seconds")