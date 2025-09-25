# AudioLDM Finetuned for soundscape generation

# 🌄 Improved Soundscape Generator 🌳 AudioLDM + Mistral LLM 🐦

This project generates realistic natural soundscapes from short text descriptions in **Spanish** or **English**, through an interactive web interface built with Gradio.
Uses the fine-tuned **AudioLDM** model and leverages the potential of the LLM **Mistral-7B** model to enrich messages for consistent results.

---

## 🍂 How it works? 🌊

- You have to write a prompt describing the soundscape you want to create.
- If the input is simple and vague, the system will improve by adding an additional level of detail using the Mistral-7B model.
- The enriched text is passed to AudioLDM to start the audio generation process.
- The web interface allows users to customize key generation parameters:

    - **Text Description 📝:** You can enter your prompt in English or Spanish  (e.g. "tormenta en la selva" / `"storm in the jungle")
    - **Guidance Scale 🍀:** Adjusts the strength of text conditioning. Higher values make audio generation more aligned with the prompt.

- Once the audio is generated, you can play it automatically, download it or generate a new one.

---

## 🌧️ Requirements 📦 

- ⚠️ The full system (AudioLDM Medium Finetuned Checkpoint + Code + VAE + CLAP) requires 13GB of disk space.
- CUDA is highly recommended


## 🌵 Instalation ⚙️ 

```shell
# 1) Create conda environment
conda create -n audioldm_finetuned python=3.10
conda activate audioldm_finetuned

# 2) Clone the repo
git clone https://github.com/anmartinez18/AudioLDM-m-Finetuned.git
cd AudioLDM-m-Finetuned

# 3) Install running environment
pip install poetry
poetry lock
poetry install

# 4) Download checkpoints (AudioLDM, VAE, CLAP, fine-tune)
poetry run get-checkpoints

```

## 👉 Setting your Mistral API Key 🔑

To use the system, you have to provide your own **Mistral API key**,  which you can obtain for free at [https://mistral.ai/](https://mistral.ai/).

You must set the variable `MISTRAL_API_KEY` before running the application:

```shell
# On Windows (CMD)
set MISTRAL_API_KEY="your_key_here"

# On Linux/Mac
export MISTRAL_API_KEY="your_key_here"

```
## 🔥 Web App 🦉

Launch the web application powered by Gradio

```shell
python app.py

# Enjoy !
```

### 🎧 Demo de audio

<audio controls>
  <source src="https://cdn.jsdelivr.net/gh/anmartinez18/AudioLDM-m-Finetuned@main/samples/Rain_from_outside_hitting_the_window.wav" type="audio/wav">
</audio>

