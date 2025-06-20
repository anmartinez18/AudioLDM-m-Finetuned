from langdetect import detect
from deep_translator import GoogleTranslator

def preprocess(text: str) -> str:
    if detect(text) == 'es':
        return GoogleTranslator(source='es', target='en').translate(text)
    elif detect(text) == 'en':
        return text
    else:
        return ("Language unknown")
    