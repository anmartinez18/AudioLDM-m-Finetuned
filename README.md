#  AudioLDM_finetuned

# 🌍 🌄 Improved Soundscape Generator 🌳s AudioLDM + Mistral LLM 🐦

This project generates realistic natural soundscapes from short text descriptions in **Spanish** or **English**, through an interactive web interface built with Gradio.
Uses the fine-tuned **AudioLDM** model and leverages the potential of the LLM **Mistral-7B** model to enrich messages for consistent results.

---

## 🍂 How it works? 🌊

- You have to write a prompt describing the soundscape you want to create.
- If the input is simple and vague, the system will improve by adding an additional level of detail using the Mistral-7B model.
- The enriched text is passed to AudioLDM to start the audio generation process.
- The web interface allows users to customize key generation parameters:

    - **Text Description 📝:** You can enter your prompt in English or Spanish
    (e.g. "tormenta en la selva" / `"storm in the jungle")

    - **Duration (seconds) ⏱️:** Choose the length of the generated soundscape, from 10 to 180 seconds (3 minutes max)

    - **DDIM Steps 🌋:** Controls the number of inference steps in the diffusion process.

    - **Guidance Scale 🍀:** Adjusts the strength of text conditioning. Higher values make audio generation more aligned with the prompt.

- Once the audio is generated, you can play it automatically, download it or generate a new one.

---

## 🌧️ Requirements 📦 



## 🌵 Instalation ⚙️ 

```shell
git clone https://github.com/anmartinez18/AudioLDM_finetuned.git
cd AudioLDM_finetuned
conda env create -f environment.yml
conda activate audioldm-finetuned-env
python app.py
```


## 🔥 Web App 🦉