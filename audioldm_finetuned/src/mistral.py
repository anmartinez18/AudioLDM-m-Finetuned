import os
import time
from mistralai import Mistral
from pathlib import Path
import sys

argumentos = sys.argv


api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("⚠️ Please set your MISTRAL_API_KEY!!!")
    sys.exit(1)

model_id = "open-mistral-7b"
client = Mistral(api_key=api_key)
path_to_template = Path(__file__).parent / "prompt_template.txt"
template = path_to_template.read_text(encoding="utf-8")

def is_brief(text: str, max_words: int = 5) -> bool:
    return len(text.strip().split()) <= max_words

def enrich_prompt(text: str) -> str:
    input_text = str(template) + str(text)
    chat_response = client.chat.complete(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": input_text,
                },
            ]
        )
    return chat_response.choices[0].message.content


if __name__=="__main__":
    print(enrich_prompt(argumentos[1:]))
    