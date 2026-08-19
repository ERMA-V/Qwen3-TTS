import argparse
from pathlib import Path

import librosa
import soundfile as sf

TARGET_SR = 24000


def main():
    parser = argparse.ArgumentParser(description="Resample a 16kHz wav file to 24kHz.")
    parser.add_argument("input", type=str, help="Path to the input 16kHz wav file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output 24kHz wav file (default: <input>_24khz.wav)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_24khz{input_path.suffix}")
    else:
        output_path = Path(args.output)

    audio, sr = librosa.load(str(input_path), sr=None, mono=True)
    if sr != TARGET_SR:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=TARGET_SR)

    sf.write(str(output_path), audio, TARGET_SR)
    print(f"Wrote {output_path} ({sr} Hz -> {TARGET_SR} Hz)")


if __name__ == "__main__":
    main()
