#!/usr/bin/env python3
import os
import requests
from tqdm import tqdm

# Zenodo URLs
URL_MODEL_FINETUNED = "https://zenodo.org/record/15676557/files/audioldm-m-full-finetuned.ckpt?download=1"
URL_CLAP = "https://zenodo.org/record/15678883/files/clap_music_speech_audioset_epoch_15_esc_89.98.pt?download=1"
URL_VAE = "https://zenodo.org/record/15678883/files/vae_mel_16k_64bins.ckpt?download=1"
URL_HIFIGAN = "https://zenodo.org/record/15678883/files/hifigan_16k_64bins.ckpt?download=1"


def download_ckpt_from_zenodo(url: str, dst_dir: str, filename: str):
    dst_path = os.path.join(dst_dir, filename)
    if os.path.exists(dst_path):
        print(f"{filename} already exists {dst_path}")
        return dst_path

    print(f"Getting {filename} . . .")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))
    with open(dst_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1_048_576):
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
    print(f"✅ {filename} ready in {dst_path}")
    return dst_path

def main():
    dst_dir = "./audioldm/ckpt/"
    dst_dir2 = "./data/checkpoints/"
    os.makedirs(dst_dir, exist_ok=True)

    download_ckpt_from_zenodo(URL_MODEL_FINETUNED, dst_dir, "audioldm-m-full-finetuned.ckpt")
    download_ckpt_from_zenodo(URL_CLAP, dst_dir2, "clap_music_speech_audioset_epoch_15_esc_89.98.pt")
    download_ckpt_from_zenodo(URL_VAE, dst_dir2, "vae_mel_16k_64bins.ckpt")
    download_ckpt_from_zenodo(URL_HIFIGAN, dst_dir2, "hifigan_16k_64bins.ckpt")

    print("\n All the checkpoints are ready!! Ready to go!!")

if __name__ == "__main__":
    main()
