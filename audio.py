from gtts import gTTS
import os

def generate_audio(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    print(f"Audio saved to {output_file}")