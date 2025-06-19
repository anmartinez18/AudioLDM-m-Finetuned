import shutil
import os

import argparse
import yaml
import torch

from audioldm.utilities.data.dataset import AudioDataset

from torch.utils.data import DataLoader
#from pytorch_lightning import seed_everything
from audioldm.utilities.model_util import instantiate_from_config

"""

def infer_long(prompt, total_secs=60, seg_secs=10, guidance_scale=3.5):
    n_segments = total_secs // seg_secs
    final_wav = []
    for i in range(n_segments):
        audio_i = latent_diffusion.generate_sample(
            ...
            name=f"infer_seg_{i}"
        )
        wav_path = os.path.join(audio_i, sorted(os.listdir(audio_i))[0])
        segment, sr = sf.read(wav_path)
        final_wav.append(segment)
    combined = np.concatenate(final_wav, axis=0)
    out_path = f"infer/combined_{prompt[:20]}.wav"
    sf.write(out_path, combined, sr)
    return out_path

"""



def infer(title, text, seconds, guidance_scale):

    assert torch.cuda.is_available(), "CUDA is not available"

    dataset_json = {
        "data": [
            {
                "wav": title,
                "caption": text
            }
        ]
    }

    """
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)
    """

    config_yaml_path = "./audioldm/config/audioldm.yaml"
    configs = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])

    log_path = configs["log_directory"]    


    dataloader_add_ons = []
    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
    )
    
    checkpoint_path = "./audioldm/ckpt/audioldm-m-full-finetuned.ckpt"

    if os.path.isfile(checkpoint_path):
        resume_from_checkpoint = checkpoint_path
        print("Resume from checkpoint:", resume_from_checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint do not exist: {checkpoint_path}")

    exp_group_name = "audioldm_finetuned"
    exp_name = "results"

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"]["ddim_sampling_steps"]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"]["n_candidates_per_samples"]

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion = latent_diffusion.cuda()

    waveform_path = latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
    )

    return waveform_path
