import gradio as gr
from src.process_text import preprocess
from src.mistral import enrich_prompt
from audioldm.infer import generate
from src.utils import  process_name

custom_css = """
   
    body {
        background-color: lightblue;	
    }

   h1 {
        text-align: center;
    }   

    p {
        font-size: 10px;
        text-align: center;
    }   
    
    .button {
        border: none;
        color: orange;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }

"""

def generate_audio_file(text, seconds, guidance):
    if text or text.strip():
        title_wav = process_name(text) + ".wav"
        text_en = preprocess(text)
        enriched_text = enrich_prompt(text_en)
        audio = generate(title_wav, enriched_text, seconds, guidance)
        final_path = audio + "/" + title_wav
        return enriched_text, final_path
    raise gr.Error(" ✏️ Please, enter a description before submitting 💥!", duration=5)
 

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎧 Prueba interfaz AudioLDM finetuned
    Esta es una prueba de AudioLDM finetuneado para ver el funcionamiento de todo esto. 
    """, elem_classes="center-text")

    with gr.Row():
        with gr.Column():
            description_input = gr.Textbox(label="Text description (English or Spanish)")
            duration = gr.Slider(10, 60, step=10, value=10, label="Duration in seconds (máx. 180 sec - 3 min)")
            guidance_scale = gr.Slider(0.1, 5.0, value=3.5, step=0.1, label="Guidance scale")
            submit_btn = gr.Button("Submit")

        with gr.Column():
            output_text = gr.Textbox(label="Resultado", interactive=False)
            output_audio = gr.Audio(type="filepath", label="🔊 Listen & Download")
    


    submit_btn.click(fn=generate_audio_file, inputs=[description_input, duration, guidance_scale], outputs=[output_text, output_audio])
    


demo.launch()
#demo.launch(share=True)


