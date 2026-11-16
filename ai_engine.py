import requests
import json

OLLAMA_URL = "http://192.168.1.41:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:14b-instruct"

def generate_story_from_idea(basic_idea: str) -> dict:
    prompt = f"""
Ești unregizor de filme și creator de conținut video AI. 
Pornind de la această idee de bază: "{basic_idea}", generează o poveste captivantă structurată în 3 scene scurte.
Returnează RĂSPUNSUL EXCLUSIV sub formă de obiect JSON valid, fără alte texte înainte sau după, având exact această structură:
{{
  "title": "Titlul clipului",
  "scenes": [
    {{"scene_number": 1, "narration": "Textul rostit pentru scena 1", "visual_prompt": "Descriere vizuala detaliata pentru scena 1"}},
    {{"scene_number": 2, "narration": "Textul rostit pentru scena 2", "visual_prompt": "Descriere vizuala detaliata pentru scena 2"}},
    {{"scene_number": 3, "narration": "Textul rostit pentru scena 3", "visual_prompt": "Descriere vizuala detaliata pentru scena 3"}}
  ]
}}
"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        result = response.json()
        return json.loads(result.get("response", "{}"))
    except Exception as e:
        print(f"Eroare la generarea poveștii: {e}")
        return {}