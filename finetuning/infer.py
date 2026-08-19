import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import argparse
import time
from transformers import BitsAndBytesConfig

model = "output/checkpoint-epoch-2"
bits = 16 # 4 or 8 or 16
device = "cuda:0"

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

tts = load_quantized(model, bits=bits, device=device)

_ = tts.generate_custom_voice(
    text="",
    language="English",
    speaker="erma-v",
)

while True:
    text = input("Enter text to generate: ")
    if text == "exit":
        break

    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(
        text=text,
        language="English",
        speaker="erma-v",
    )
    sf.write("infer.wav", wavs[0], sr)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.2f} seconds")