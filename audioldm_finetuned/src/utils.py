import re
import soundfile as sf

def save_wav(file: str, audio, sample_rate: int = 16000):
    with open(file, "wb") as f:
        sf.write(file, audio, sample_rate)


def process_name(text: str) -> str:
    text = text.replace('"', '').replace("'", "")
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', text).strip('_')
