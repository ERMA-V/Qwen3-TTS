import json
import librosa
import soundfile as sf
from pathlib import Path

with open("./dataset/original/metadata.list", "r") as f_in, open("./dataset/qwen/dataset.jsonl", "w") as f_out:
    for line in f_in:
        # audio/00000.wav|melo_ermav01|EN|okay.
        audio_path, speaker_id, language, text = line.strip().split("|")
        original_audio_path = "./dataset/original/" + audio_path
        new_audio_path = "./dataset/qwen/" + audio_path
        if not Path(new_audio_path).exists():
            audio, sr = librosa.load(str(original_audio_path), sr=None, mono=True)
            if sr != 24000:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=24000)
            sf.write(str(new_audio_path), audio, 24000)
        # {"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
        f_out.write(json.dumps({"audio": new_audio_path, "text": text, "ref_audio": "./dataset/qwen/reference.wav"}) + "\n")