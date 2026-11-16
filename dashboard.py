import os
os.environ["LD_LIBRARY_PATH"] = "/root/playwright_libs/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
import re
import sys
import time
import uuid
import json
import shutil
import psutil
import subprocess
import numpy as np
import soundfile as sf
import streamlit as st
import requests
from datetime import datetime, timedelta, timezone
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel
from playwright.sync_api import sync_playwright
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload

st.set_page_config(page_title="AI Studio V13 - RTX 3060 Optimized & Auto-Scheduler", layout="wide", page_icon="🎬")

VOCI_KOKORO = {
    "Bella + Heart Blend (Warm Storyteller / Kids)": {"id": "blend_bella_heart", "lang": "en-us"},
    "Bella (Cheerful Female / Young Character)": {"id": "af_bella", "lang": "en-us"},
    "Heart (Sweet Child / Gentle Character)": {"id": "af_heart", "lang": "en-us"},
    "Adam (Deep Male / Papa Bear / Narrator)": {"id": "am_adam", "lang": "en-us"},
    "Emma (British Female / Wise Teacher)": {"id": "bf_emma", "lang": "en-gb"},
    "George (British Male / Grandpa / Storyteller)": {"id": "bm_george", "lang": "en-gb"}
}

MAPARE_VOCE_ROL = {
    "narrator": "blend_bella_heart",
    "child_or_cute": "af_heart",
    "female_cheerful": "af_bella",
    "male_deep": "am_adam",
    "wise_british": "bf_emma"
}

PALETE_DISPONIBILE = [
    "Enchanted Forest & Nature (Lush Green, Warm Wood, Sunlit Gold)",
    "Sunny Kids & Classroom (Sky Blue, Vibrant Yellow, Coral Red)",
    "Vibrant City & Town (Modern Blue, Sunset Orange, Warm Streetlight)",
    "Cosmic Space & Mystery (Deep Indigo, Neon Cyan, Magenta)",
    "Ocean & Underwater (Aqua Marine, Coral Pink, Sandy Gold)"
]

FISIER_CAPITOLE = "/root/capitole_studio.json"
FISIER_AUTOPILOT = "/root/autopilot_config.json"
DIR_ARHIVA_EPISOADE = "/root/episoade_arhiva"

@st.cache_resource
def incarca_motor_kokoro():
    # Activare accelerare CUDA pentru Kokoro pe RTX 3060
    return Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

@st.cache_resource
def incarca_motor_whisper():
    # Rulare Whisper direct pe RTX 3060 cu float16 pentru viteză maximă
    return WhisperModel("small.en", device="cuda", compute_type="float16")

def curata_episoad_vechi_7_zile():
    if not os.path.exists(DIR_ARHIVA_EPISOADE):
        os.makedirs(DIR_ARHIVA_EPISOADE, exist_ok=True)
        return
    acum = time.time()
    limita_secunde = 7 * 24 * 3600
    for nume_fold in os.listdir(DIR_ARHIVA_EPISOADE):
        cale_fold = os.path.join(DIR_ARHIVA_EPISOADE, nume_fold)
        if os.path.isdir(cale_fold):
            try:
                if (acum - os.path.getmtime(cale_fold)) > limita_secunde:
                    shutil.rmtree(cale_fold)
            except Exception:
                pass

def incarca_baza_capitole():
    if os.path.exists(FISIER_CAPITOLE):
        try:
            with open(FISIER_CAPITOLE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "capitole" in data and isinstance(data["capitole"], dict):
                    return data
        except Exception:
            pass
    cap_id = "cap_story_1"
    default_db = {
        "capitol_activ": cap_id,
        "capitole": {
            cap_id: {
                "id": cap_id,
                "nume": "Forest Bears & Friends",
                "descriere_tema": "Animated stories for children about two friendly bears exploring the forest, learning life lessons, the alphabet, and nature secrets through fun dialogues",
                "personaje_si_lume": "Barnaby (a big friendly brown bear with a deep warm voice) and Pip (a curious little bear cub), living in a lush animated forest with talking animals",
                "paleta_culori": PALETE_DISPONIBILE[0],
                "creat_la": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "episoade": []
            }
        }
    }
    salveaza_baza_capitole(default_db)
    return default_db

def salveaza_baza_capitole(db):
    with open(FISIER_CAPITOLE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def incarca_autopilot():
    default_ap = {
        "activ": False, "interval_ore": 12.0, "model": "qwen2.5:14b", "nr_scene": 8, "vizibilitate": "public", "ultima_rulare": "", "urmatorul_run_ts": 0
    }
    if os.path.exists(FISIER_AUTOPILOT):
        try:
            with open(FISIER_AUTOPILOT, "r", encoding="utf-8") as f:
                default_ap.update(json.load(f))
        except Exception:
            pass
    return default_ap

def salveaza_autopilot(ap):
    with open(FISIER_AUTOPILOT, "w", encoding="utf-8") as f:
        json.dump(ap, f, ensure_ascii=False, indent=2)

def apel_ollama_json(model, prompt, temp=0.72):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": temp, "num_predict": 3000}
        },
        timeout=600
    )
    r.raise_for_status()
    return json.loads(r.json().get("response", "{}").strip())

def curata_text(text):
    t = str(text).strip()
    t = re.sub(r"^(Sentence \d+|First|Second|Third|Fourth|Fifth|Sixth|Narrator|Speaker \d+)\s*[:,.-]\s*", "", t, flags=re.IGNORECASE)
    return t.strip()

def inventeaza_univers_capitol(model, idee_utilizator):
    prompt = f"""You are an Expert Creative Showrunner. 
User idea / seed: "{idee_utilizator}"

Develop this series staying 100% faithful to the user's core vision. Expand it into a coherent, rich, and engaging universe.
Output valid JSON with keys:
- "chapter_name": Short catchy series title (max 5 words)
- "expanded_premise": Detailed description of what this series is about and how episodes progress
- "characters_and_world": Detailed description of recurring characters, their unique personalities, and the setting
- "color_palette": Choose ONE from: 
{json.dumps(PALETE_DISPONIBILE)}
"""
    return apel_ollama_json(model, prompt)

def genereaza_plan_adaptiv(model, capitol, nr_scene, instructiune_sau_script=""):
    istoric_ep = capitol.get("episoade", [])
    ep_nr = len(istoric_ep) + 1
    context_istoric = ""
    if istoric_ep:
        rez = " | ".join([f"Ep.{e['ep']} ({e['titlu']}): {e.get('rezumat', '')}" for e in istoric_ep[-5:]])
        context_istoric = f"PREVIOUS EPISODES: {rez}.\nContinue logically from Episode {ep_nr-1}. Maintain narrative consistency and theme."

    if len(instructiune_sau_script.strip()) > 120:
        cerinta = f"The user provided this custom script/story:\n\"{instructiune_sau_script.strip()}\"\nBreak this exact story into {nr_scene} sequential scenes!"
    elif instructiune_sau_script.strip():
        cerinta = f"Specific topic/request for Episode {ep_nr}: \"{instructiune_sau_script.strip()}\""
    else:
        cerinta = f"Autonomously invent the next thrilling episode (Episode {ep_nr}) maintaining absolute continuity with the series theme."

    prompt = f"""You are directing Episode {ep_nr} of "{capitol['nume']}".
Series Premise: {capitol['descriere_tema']}
Characters & World: {capitol['personaje_si_lume']}
{context_istoric}
{cerinta}

Divide Episode {ep_nr} into EXACTLY {nr_scene} sequential scenes.
Output valid JSON with:
- "episode_summary": 1-sentence summary of this episode's plot
- "scene_topics": A JSON array of EXACTLY {nr_scene} strings describing step-by-step actions for each scene.
"""
    data = apel_ollama_json(model, prompt)
    topice = data.get("scene_topics", [])
    while len(topice) < nr_scene:
        topice.append(f"Scene {len(topice)+1}: The adventure continues in {capitol['nume']}")
    return data.get("episode_summary", ""), topice[:nr_scene]

def genereaza_scena_dinamica_universala(model, capitol, idx, total_scene, topic_scena):
    prompt = f"""You are animating Scene {idx} of {total_scene} for "{capitol['nume']}".
World & Characters: {capitol['personaje_si_lume']}
What happens in Scene {idx}: "{topic_scena}"

Design a vivid 2D animated scene matching this setting:
1. "scene_mode": Choose ONE: "story_stage", "magic_showcase", "infographic_cards"
2. "environment": Choose ONE: "forest", "city", "classroom", "ocean", "space"
3. "time_of_day": Choose ONE: "sunny_day", "golden_sunset", "starry_night"
4. "weather_effect": Choose ONE: "sunbeams", "fireflies", "gentle_snow", "rain", "floating_leaves"
5. "headline": Short scene banner (max 32 chars)
6. "central_prop": Central object/symbol (e.g. "🍯 Honey Jar", "✨ Magic Star", "🚗 Red Car")
7. "actors": Array of 1 to 3 characters:
- "id": "actor_1", "actor_2", or "actor_3"
- "name": Character name
- "avatar_type": "bear_brown", "bear_cub", "fox_cute", "bunny", "owl_wise", "kid_boy", "kid_girl", "teacher_woman", "robot_friend"
- "emoji_fallback": Matching emoji
- "voice_role": "male_deep", "child_or_cute", "female_cheerful", "wise_british", "narrator"
8. "dialogue_lines": Array of 4 to 6 natural spoken lines (total 60-80 words). Each item has:
- "speaker_id", "speaker_name", "voice_role", "text" (English), "speech_bubble" (short 2-5 words popup)

Output ONLY valid JSON.
"""
    data = apel_ollama_json(model, prompt, temp=0.74)
    actors = data.get("actors", [])
    if not actors:
        actors = [
            {"id": "actor_1", "name": "Barnaby", "avatar_type": "bear_brown", "emoji_fallback": "🐻", "voice_role": "male_deep"},
            {"id": "actor_2", "name": "Pip", "avatar_type": "bear_cub", "emoji_fallback": "🧸", "voice_role": "child_or_cute"}
        ]
        data["actors"] = actors
    return data

def genereaza_seo_complet_youtube(model, capitol, rezumat_ep, script_integral, timestamps_list):
    ts_text = "\n".join(timestamps_list)
    prompt = f"""You are a YouTube SEO Expert. Generate metadata for "{capitol['nume']}".
Episode Summary: {rezumat_ep}
Script Excerpt: {script_integral[:1200]}

Output valid JSON with:
- "seo_title": Catchy YouTube title (60-90 chars)
- "seo_description": Rich 3-paragraph SEO description with hashtags
- "seo_tags": Array of 15 high-volume search tags
"""
    seo = apel_ollama_json(model, prompt, temp=0.65)
    titlu = seo.get("seo_title", f"{capitol['nume']} - {rezumat_ep[:50]}")[:98]
    desc_baza = seo.get("seo_description", rezumat_ep)
    desc_finala = f"{desc_baza}\n\n⏱️ VIDEO CHAPTERS:\n{ts_text}\n\n🔔 Subscribe for more adventures!"
    tags = seo.get("seo_tags", ["animation", "kids education", "stories for kids", "learning"])
    return titlu, desc_finala, tags

def obtine_voce_kokoro(kokoro, voice_id):
    if voice_id == "blend_bella_heart":
        v1 = kokoro.get_voice_style("af_bella")
        v2 = kokoro.get_voice_style("af_heart")
        return (v1 * 0.65) + (v2 * 0.35)
    return voice_id

def sintetizeaza_dialog_multivoce(dialogue_lines, default_voice_id, fisier_wav, bg_vol=0.05):
    kokoro = incarca_motor_kokoro()
    bucati = []
    timeline_vorbitori = []
    sr_final = 24000
    timp_curent = 0.0

    for l in dialogue_lines:
        txt = l.get("text", "").replace('"', '').replace("\n", " ").strip()
        if not txt:
            continue
        rol = l.get("voice_role", "narrator")
        v_id = MAPARE_VOCE_ROL.get(rol, default_voice_id)
        lang_c = "en-gb" if v_id.startswith("b") else "en-us"
        stil_voce = obtine_voce_kokoro(kokoro, v_id)

        spd = 0.91 if rol == "child_or_cute" else (0.93 if rol == "male_deep" else 0.95)
        samples, sr = kokoro.create(txt, voice=stil_voce, speed=spd, lang=lang_c)
        sr_final = sr

        durata_replica = len(samples) / float(sr)
        pauza_sec = 0.42
        pauza_samples = np.zeros(int(sr * pauza_sec), dtype=np.float32)

        timeline_vorbitori.append({
            "start": round(timp_curent, 2),
            "end": round(timp_curent + durata_replica, 2),
            "speaker_id": l.get("speaker_id", "actor_1"),
            "speaker_name": l.get("speaker_name", ""),
            "bubble": l.get("speech_bubble", txt[:28] + "...")
        })

        timp_curent += durata_replica + pauza_sec
        bucati.append(samples)
        bucati.append(pauza_samples)

    audio_voce = np.concatenate(bucati) if bucati else np.zeros(24000, dtype=np.float32)
    max_v = np.max(np.abs(audio_voce))
    if max_v > 0:
        audio_voce = (audio_voce / max_v) * 0.90

    if bg_vol > 0 and os.path.exists("/root/bg_music.wav"):
        try:
            bg_data, _ = sf.read("/root/bg_music.wav", dtype="float32")
            if bg_data.ndim > 1:
                bg_data = bg_data.mean(axis=1)
            if len(bg_data) < len(audio_voce):
                bg_data = np.tile(bg_data, int(len(audio_voce) // len(bg_data)) + 2)
            audio_voce = np.clip(audio_voce + bg_data[:len(audio_voce)] * bg_vol, -0.98, 0.98)
        except Exception:
            pass

    sf.write(fisier_wav, audio_voce, sr_final)
    durata_totala = len(audio_voce) / float(sr_final)
    return durata_totala, timeline_vorbitori

def deseneaza_svg_personaj(avatar_type, emoji_fb):
    culori_corp = {
        "bear_brown": ("#8D5524", "#C68642", "🐻"),
        "bear_cub": ("#A66E38", "#E0AC69", "🧸"),
        "fox_cute": ("#E65100", "#FFE0B2", "🦊"),
        "bunny": ("#F5F5F5", "#FFCDD2", "🐰"),
        "owl_wise": ("#5D4037", "#D7CCC8", "🦉"),
        "kid_boy": ("#FFCCBC", "#29B6F6", "👦"),
        "kid_girl": ("#FFCCBC", "#EC407A", "👧"),
        "teacher_woman": ("#FFE0B2", "#FFD54F", "👩‍🏫"),
        "robot_friend": ("#78909C", "#00E5FF", "🤖")
    }
    c1, c2, icon_top = culori_corp.get(avatar_type, ("#FFB74D", "#FFF3E0", emoji_fb or "🌟"))

    return f"""
    <svg viewBox="0 0 260 300" width="250" height="290" class="char-svg">
      <ellipse cx="130" cy="282" rx="75" ry="14" fill="rgba(0,0,0,0.28)"/>
      <circle cx="65" cy="65" r="34" fill="{c1}" stroke="#ffffff" stroke-width="5"/>
      <circle cx="65" cy="65" r="18" fill="{c2}"/>
      <circle cx="195" cy="65" r="34" fill="{c1}" stroke="#ffffff" stroke-width="5"/>
      <circle cx="195" cy="65" r="18" fill="{c2}"/>
      <rect x="65" y="165" width="130" height="110" rx="55" fill="{c1}" stroke="#ffffff" stroke-width="5"/>
      <ellipse cx="130" cy="220" rx="42" ry="38" fill="{c2}"/>
      <ellipse class="char-arm-left" cx="52" cy="205" rx="22" ry="38" fill="{c1}" stroke="#ffffff" stroke-width="4"/>
      <ellipse class="char-arm-right" cx="208" cy="205" rx="22" ry="38" fill="{c1}" stroke="#ffffff" stroke-width="4"/>
      <circle cx="130" cy="115" r="72" fill="{c1}" stroke="#ffffff" stroke-width="6"/>
      <ellipse cx="130" cy="132" rx="44" ry="34" fill="{c2}"/>
      <g class="char-eyes">
        <circle cx="102" cy="98" r="14" fill="#ffffff"/>
        <circle cx="158" cy="98" r="14" fill="#ffffff"/>
        <circle cx="104" cy="98" r="7" fill="#1a1a2e"/>
        <circle cx="156" cy="98" r="7" fill="#1a1a2e"/>
      </g>
      <ellipse cx="130" cy="122" rx="13" ry="9" fill="#2d1e18"/>
      <ellipse class="char-mouth" cx="130" cy="145" rx="20" ry="6" fill="#880E4F" stroke="#ffffff" stroke-width="2"/>
    </svg>
    """

def construieste_decor_svg(env_type, time_of_day):
    cer = {
        "sunny_day": ("#38BDF8", "#BAE6FD", "#FDE047"),
        "golden_sunset": ("#F97316", "#FDE047", "#FB923C"),
        "starry_night": ("#0F172A", "#1E1B4B", "#FEF08A")
    }.get(time_of_day, ("#38BDF8", "#BAE6FD", "#FDE047"))

    if env_type == "forest":
        return cer, """
        <polygon points="0,780 350,420 750,780" fill="#2D6A4F" opacity="0.6"/>
        <polygon points="550,780 1050,360 1550,780" fill="#40916C" opacity="0.65"/>
        <rect x="0" y="820" width="1920" height="260" fill="#2D6A4F"/>
        <g class="tree-sway">
          <rect x="155" y="620" width="36" height="210" fill="#5C3A21"/>
          <circle cx="173" cy="560" r="115" fill="#1B4332"/>
        </g>
        """
    elif env_type == "city":
        return cer, """
        <rect x="80" y="340" width="220" height="500" rx="12" fill="#334155"/>
        <rect x="330" y="420" width="190" height="420" rx="12" fill="#475569"/>
        <rect x="0" y="860" width="1920" height="220" fill="#334155"/>
        """
    else:
        return ("#38BDF8", "#BAE6FD", "#FDE047"), "<rect x='0' y='800' width='1920' height='280' fill='#0284C7'/>"

def construieste_scena_html_dinamica(date_scena, idx, total_scene, durata_sec, timeline_vorbitori):
    env_type = date_scena.get("environment", "forest")
    time_of_day = date_scena.get("time_of_day", "sunny_day")
    headline = str(date_scena.get("headline", f"Scene {idx}"))[:42]
    central_prop = str(date_scena.get("central_prop", "✨"))[:28]
    actors = date_scena.get("actors", [])[:3]

    cer_culori, svg_decor = construieste_decor_svg(env_type, time_of_day)
    bg_top, bg_bot, sun_col = cer_culori

    actors_html = ""
    for a in actors:
        aid = a.get("id", "actor_1")
        aname = a.get("name", "Friend")[:18]
        svg_char = deseneaza_svg_personaj(a.get("avatar_type", "bear_brown"), a.get("emoji_fallback", "🐻"))
        actors_html += f"""
        <div class="actor-unit" id="{aid}">
          <div class="speech-bubble" id="bubble_{aid}">Hello!</div>
          <div class="char-body">{svg_char}</div>
          <div class="actor-name">{aname}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1920px; height: 1080px; overflow: hidden;
    background: linear-gradient(180deg, {bg_top} 0%, {bg_bot} 78%);
    font-family: 'DejaVu Sans', 'Segoe UI', sans-serif; position: relative;
  }}
  svg.world-bg {{ position: absolute; top: 0; left: 0; width: 1920px; height: 1080px; z-index: 1; }}
  .sun-orb {{
    position: absolute; top: 75px; right: 140px; width: 135px; height: 135px;
    border-radius: 50%; background: {sun_col}; box-shadow: 0 0 80px {sun_col}; z-index: 1;
  }}
  .top-hud {{
    position: relative; z-index: 5; padding: 38px 70px;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .scene-title {{
    background: rgba(15, 23, 42, 0.82); color: #ffffff;
    padding: 18px 36px; border-radius: 28px; font-size: 42px; font-weight: 900;
    border: 4px solid #FACC15; box-shadow: 0 15px 35px rgba(0,0,0,0.3);
  }}
  .prop-showcase {{
    position: absolute; top: 185px; left: 50%; transform: translateX(-50%);
    z-index: 4; background: rgba(255, 255, 255, 0.94);
    border: 6px solid #F59E0B; border-radius: 42px;
    padding: 22px 55px; font-size: 64px; font-weight: 900; color: #0F172A;
    box-shadow: 0 25px 60px rgba(0,0,0,0.32);
  }}
  .stage-actors {{
    position: absolute; bottom: 85px; left: 0; width: 1920px;
    z-index: 6; display: flex; justify-content: space-around; align-items: flex-end;
    padding: 0 180px;
  }}
  .actor-unit {{ display: flex; flex-direction: column; align-items: center; }}
  .actor-unit.talking {{ filter: drop-shadow(0 0 28px rgba(250, 204, 21, 0.95)); transform: scale(1.05); }}
  .speech-bubble {{
    background: #ffffff; color: #0F172A; font-size: 32px; font-weight: 900;
    padding: 18px 30px; border-radius: 28px; border: 5px solid #F59E0B;
    margin-bottom: 18px; max-width: 440px; text-align: center;
    opacity: 0; transform: scale(0.5); transition: all 0.2s ease;
  }}
  .speech-bubble.visible {{ opacity: 1; transform: scale(1); }}
  .actor-name {{
    margin-top: -8px; background: #0F172A; color: #FACC15;
    padding: 8px 26px; border-radius: 20px; font-size: 26px; font-weight: 800; border: 3px solid #ffffff;
  }}
</style>
</head>
<body>
  <div class="sun-orb"></div>
  <svg class="world-bg" viewBox="0 0 1920 1080" preserveAspectRatio="none">{svg_decor}</svg>
  <div class="top-hud">
    <div class="scene-title">🎬 {headline}</div>
    <div class="scene-title" style="font-size:28px;">Scene {idx}/{total_scene}</div>
  </div>
  <div class="prop-showcase">{central_prop}</div>
  <div class="stage-actors">{actors_html}</div>
  <script>
    const timeline = {json.dumps(timeline_vorbitori)};
    const startTime = performance.now();
    function updateActors() {{
      const elapsed = (performance.now() - startTime) / 1000.0;
      document.querySelectorAll('.actor-unit').forEach(el => el.classList.remove('talking'));
      document.querySelectorAll('.speech-bubble').forEach(b => b.classList.remove('visible'));
      for (let item of timeline) {{
        if (elapsed >= item.start && elapsed <= item.end) {{
          let target = document.getElementById(item.speaker_id);
          if (target) {{
            target.classList.add('talking');
            let b = target.querySelector('.speech-bubble');
            if (b) {{ b.textContent = item.bubble; b.classList.add('visible'); }}
          }}
          break;
        }}
      }}
      requestAnimationFrame(updateActors);
    }}
    requestAnimationFrame(updateActors);
  </script>
</body>
</html>"""

def randeaza_scena_html5_video(html_content, fisier_audio, durata_sec, idx, dir_episod, fisier_mp4_scena):
    fisier_html = os.path.join(dir_episod, f"scena_{idx}.html")
    dir_video_temp = os.path.join(dir_episod, f"tmp_vid_{idx}")
    os.makedirs(dir_video_temp, exist_ok=True)

    with open(fisier_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=dir_video_temp,
            record_video_size={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        page.goto(f"file://{fisier_html}")
        time.sleep(durata_sec)
        video_path_webm = page.video.path()
        context.close()
        browser.close()

    # ACUM FOLOSIM NVENC HARDWARE PE RTX 3060 (-c:v h264_nvenc) PENTRU VITEZĂ MAXIMĂ
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path_webm, "-i", fisier_audio,
        "-t", f"{durata_sec:.2f}", "-c:v", "h264_nvenc", "-preset", "p4",
        "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "192k",
        fisier_mp4_scena
    ], check=True, capture_output=True)

def concateneaza_scene_ffmpeg(lista_mp4, fisier_iesire, dir_episod):
    lista_txt = os.path.join(dir_episod, "lista_concat.txt")
    with open(lista_txt, "w", encoding="utf-8") as f:
        for mp4 in lista_mp4:
            f.write(f"file '{os.path.abspath(mp4)}'\n")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", lista_txt, "-c", "copy", fisier_iesire
    ], check=True, capture_output=True)

def format_timp_ass(secunde):
    ore = int(secunde // 3600)
    minute = int((secunde % 3600) // 60)
    sec = secunde % 60
    return f"{ore}:{minute:02d}:{sec:05.2f}"

def format_timp_youtube(secunde):
    m = int(secunde // 60)
    s = int(secunde % 60)
    return f"{m:02d}:{s:02d}"

def adauga_subtitrari_whisper(video_intrare, video_iesire, dir_episod):
    whisper = incarca_motor_whisper()
    segments, _ = whisper.transcribe(video_intrare, word_timestamps=True)

    header_ass = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,58,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,80,80,22,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    linii_ass = [header_ass]
    grup = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            grup.append(w)
            if len(grup) >= 5 or w.word.strip().endswith((".", "!", "?", ",")):
                linii_ass.append(f"Dialogue: 0,{format_timp_ass(grup[0].start)},{format_timp_ass(grup[-1].end)},Default,,0,0,0,,{' '.join([x.word.strip() for x in grup])}\n")
                grup = []
    if grup:
        linii_ass.append(f"Dialogue: 0,{format_timp_ass(grup[0].start)},{format_timp_ass(grup[-1].end)},Default,,0,0,0,,{' '.join([x.word.strip() for x in grup])}\n")

    sub_path = os.path.join(dir_episod, "subtitrare.ass")
    with open(sub_path, "w", encoding="utf-8") as f:
        f.writelines(linii_ass)

    subprocess.run([
        "ffmpeg", "-y", "-i", video_intrare, "-vf", f"ass={sub_path}",
        "-c:v", "h264_nvenc", "-preset", "p4", "-c:a", "copy", video_iesire
    ], check=True, capture_output=True)

def upload_youtube_seo(video_path, titlu, descriere, tags_list, cat_id="27", vizibilitate="public", publish_at_iso=None):
    creds = Credentials.from_authorized_user_file('/root/token.json', ['https://www.googleapis.com/auth/youtube.upload'])
    youtube = build('youtube', 'v3', credentials=creds)

    status_dict = {'privacyStatus': vizibilitate, 'selfDeclaredMadeForKids': False}
    if publish_at_iso:
        status_dict['privacyStatus'] = 'private'
        status_dict['publishAt'] = publish_at_iso

    body = {
        'snippet': {'title': titlu[:100], 'description': descriere, 'tags': tags_list[:15], 'categoryId': cat_id},
        'status': status_dict
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    return youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()

def executa_productie_completa_episod(model_ollama, nr_scene, voce_default_id, bg_vol, subtitrari_active, instructiune_ep="", auto_upload=False, viz_auto="public", publish_at_iso=None, ui_status=None):
    curata_episoad_vechi_7_zile()

    db = incarca_baza_capitole()
    cap_id = db["capitol_activ"]
    capitol = db["capitole"][cap_id]
    ep_urmator = len(capitol["episoade"]) + 1

    os.makedirs(DIR_ARHIVA_EPISOADE, exist_ok=True)
    timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_episod = os.path.join(DIR_ARHIVA_EPISOADE, f"ep_{ep_urmator}_{timestamp_folder}")
    os.makedirs(dir_episod, exist_ok=True)

    if ui_status:
        ui_status.write(f"📁 Folder creat: `{dir_episod}`\n🧠 Pasul 1: Planificare poveste cu modelul **{model_ollama}** pe RTX 3060...")
    rezumat_ep, lista_topice = genereaza_plan_adaptiv(model_ollama, capitol, nr_scene, instructiune_ep)

    fisiere_mp4_scene = []
    script_complet = []
    timestamps_yt = []
    timp_acumulat = 0.0

    for idx, topic_scena in enumerate(lista_topice, start=1):
        if ui_status:
            ui_status.write(f"🎙️🌲 Scena {idx}/{len(lista_topice)}: Generare audio Kokoro-CUDA & Randare NVENC...")
        date_scena = genereaza_scena_dinamica_universala(model_ollama, capitol, idx, len(lista_topice), topic_scena)
        d_lines = date_scena.get("dialogue_lines", [])

        ts_label = format_timp_youtube(timp_acumulat)
        timestamps_yt.append(f"{ts_label} - {date_scena.get('headline', f'Part {idx}')}")

        text_scena = " ".join([f"{l.get('speaker_name', '')}: {l.get('text', '')}" for l in d_lines])
        script_complet.append(f"[Scene {idx}] {text_scena}")

        f_wav = os.path.join(dir_episod, f"scena_{idx}.wav")
        durata_sec, timeline_vorbitori = sintetizeaza_dialog_multivoce(d_lines, voce_default_id, f_wav, bg_vol)
        timp_acumulat += durata_sec

        html_cod = construieste_scena_html_dinamica(date_scena, idx, len(lista_topice), durata_sec, timeline_vorbitori)
        f_mp4_sc = os.path.join(dir_episod, f"scena_{idx}_vec.mp4")
        randeaza_scena_html5_video(html_cod, f_wav, durata_sec, idx, dir_episod, f_mp4_sc)
        fisiere_mp4_scene.append(f_mp4_sc)

    if ui_status:
        ui_status.write("🎞️ Pasul 3: Îmbinare finală cu accelerare hardware NVENC...")
    
    fisier_brut = os.path.join(dir_episod, "rezultat_fara_sub.mp4") if subtitrari_active else "/root/rezultat.mp4"
    concateneaza_scene_ffmpeg(fisiere_mp4_scene, fisier_brut, dir_episod)

    if subtitrari_active:
        try:
            adauga_subtitrari_whisper(fisier_brut, "/root/rezultat.mp4", dir_episod)
        except Exception:
            os.replace(fisier_brut, "/root/rezultat.mp4")

    shutil.copy("/root/rezultat.mp4", os.path.join(dir_episod, "video_final.mp4"))

    script_unit = "\n\n".join(script_complet)
    titlu_seo, desc_seo, tags_seo = genereaza_seo_complet_youtube(model_ollama, capitol, rezumat_ep, script_unit, timestamps_yt)

    capitol["episoade"].append({
        "ep": ep_urmator, "titlu": titlu_seo, "descriere": desc_seo, "tags": tags_seo,
        "rezumat": rezumat_ep, "script": script_unit, "folder_arhiva": dir_episod,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    salveaza_baza_capitole(db)

    yt_id = None
    if auto_upload and os.path.exists("/root/token.json"):
        ryt = upload_youtube_seo("/root/rezultat.mp4", titlu_seo, desc_seo, tags_seo, "27", viz_auto, publish_at_iso)
        yt_id = ryt.get("id")

    return titlu_seo, desc_seo, tags_seo, yt_id

# --- INTERFATA STREAMLIT ---
db = incarca_baza_capitole()
cap_activ_id = db.get("capitol_activ", list(db["capitole"].keys())[0])
capitol_curent = db["capitole"][cap_activ_id]
ap_cfg = incarca_autopilot()

st.title("🎬 AI Studio V13 — RTX 3060 GPU Accelerated & Smart AI")

with st.sidebar:
    st.header("📂 1. Capitol / Serie Activă")
    lista_id = list(db["capitole"].keys())
    id_selectat = st.selectbox(
        "Comută Capitolul:", options=lista_id, index=lista_id.index(cap_activ_id),
        format_func=lambda cid: f"📁 {db['capitole'][cid]['nume']} ({len(db['capitole'][cid]['episoade'])} ep.)"
    )
    if id_selectat != cap_activ_id:
        db["capitol_activ"] = id_selectat
        salveaza_baza_capitole(db)
        st.rerun()

    st.divider()
    st.header("🧠 2. Creier AI (Recomandat: Qwen 2.5 14B)")
    # Aici poți selecta modelul avansat care rulează pe GPU
    model_ollama = st.selectbox("Model LLM:", ["qwen2.5:14b", "llama3:8b", "gemma2:9b", "mistral"], index=0)
    nr_scene = st.slider("Număr de Scene:", min_value=4, max_value=24, value=8)

    st.header("🎙️ 3. Voce Principală & Sunet")
    voce_label = st.selectbox("Voce Implicită:", list(VOCI_KOKORO.keys()))
    voce_cfg = VOCI_KOKORO[voce_label]
    bg_vol = st.slider("Volum Muzică Fundal:", 0.0, 0.15, 0.05, step=0.01)
    subtitrari_active = st.checkbox("📝 Subtitrări Sincronizate (Whisper CUDA)", value=True)

tab_productie, tab_scheduler, tab_capitole, tab_youtube = st.tabs([
    f"🚀 1. Studio Animație ('{capitol_curent['nume']}')",
    "⏰ 2. AUTO-SCHEDULER & Curățare 7 Zile",
    "📚 3. MANAGER CAPITOLE",
    "📤 4. Upload YouTube"
])

with tab_productie:
    ep_urmator = len(capitol_curent["episoade"]) + 1
    st.subheader(f"🎬 Capitol Activ: **{capitol_curent['nume']}** | Urmează **Episodul {ep_urmator}**")

    col_st, col_dr = st.columns([3, 2])
    with col_st:
        edit_tema = st.text_area("Tema & Regula Capitolului:", value=capitol_curent["descriere_tema"], height=70)
        edit_lume = st.text_area("Personajele & Decorul:", value=capitol_curent["personaje_si_lume"], height=70)
        instructiune_ep = st.text_area("Scrie o idee simplă (AI-ul o va extinde creativ menținând coerența):", placeholder="Ex: Urșii descoperă o peșteră secretă cu cristale magice...", height=95)

    with col_dr:
        st.info("""**🚀 RTX 3060 Acceleration Active:**
- **LLM:** Rulat pe GPU prin Ollama.
- **Kokoro & Whisper:** Rulate nativ pe CUDA.
- **FFmpeg:** Randare hardware NVENC activată.
- **Auto-Cleanup:** Folderele mai vechi de 7 zile se șterg automat.""")
        auto_upload = st.checkbox("📤 Urcă AUTOMAT pe YouTube cu SEO", value=False)
        viz_auto = st.selectbox("Vizibilitate Auto-Upload:", ["public", "private", "unlisted"])

    if st.button(f"⚡ GENEREAZĂ EPISODUL {ep_urmator} CU RTX 3060", type="primary", use_container_width=True):
        capitol_curent["descriere_tema"] = edit_tema
        capitol_curent["personaje_si_lume"] = edit_lume
        salveaza_baza_capitole(db)

        with st.status(f"🎬 Se produce Episodul {ep_urmator}...", expanded=True) as status:
            try:
                titlu_seo, desc_seo, tags_seo, yt_id = executa_productie_completa_episod(
                    model_ollama=model_ollama, nr_scene=nr_scene, voce_default_id=voce_cfg["id"],
                    bg_vol=bg_vol, subtitrari_active=subtitrari_active, instructiune_ep=instructiune_ep,
                    auto_upload=auto_upload, viz_auto=viz_auto, ui_status=status
                )
                st.session_state["ultimul_titlu"] = titlu_seo
                st.success(f"**Titlu Generat:** {titlu_seo}")
                if yt_id:
                    st.success(f"✅ Publicat pe YouTube: https://youtu.be/{yt_id}")
                status.update(label=f"✅ Episodul {ep_urmator} gata!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Eroare: {e}")

    if os.path.exists("/root/rezultat.mp4"):
        st.video("/root/rezultat.mp4")

with tab_scheduler:
    st.subheader("⏰ Auto-Pilot 24/7 & Curățare 7 Zile")
    c_ap1, c_ap2 = st.columns([3, 2])
    with c_ap1:
        interval_ales = st.number_input("Interval (ore):", min_value=0.5, max_value=168.0, value=float(ap_cfg.get("interval_ore", 12.0)), step=0.5)
        if st.button("🟢 PORNEȘTE AUTO-PILOTUL", type="primary", use_container_width=True):
            subprocess.Popen(["python3", "/root/dashboard.py", "--daemon"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ap_cfg["activ"] = True
            ap_cfg["interval_ore"] = interval_ales
            ap_cfg["model"] = model_ollama
            ap_cfg["nr_scene"] = nr_scene
            ap_cfg["urmatorul_run_ts"] = time.time() + 10
            salveaza_autopilot(ap_cfg)
            st.success("Auto-Pilot pornit în fundal!")
            st.rerun()
        if st.button("🔴 OPREȘTE AUTO-PILOTUL", use_container_width=True):
            ap_cfg["activ"] = False
            salveaza_autopilot(ap_cfg)
            subprocess.run(["pkill", "-f", "dashboard.py --daemon"])
            st.warning("Oprit.")
            st.rerun()
    with c_ap2:
        st.metric("Stare", "🟢 ACTIV" if ap_cfg.get("activ") else "🔴 OPRIT")

with tab_capitole:
    st.subheader("➕ Adaugă Capitol Nou")
    idee_scurta = st.text_area("Scrie o idee simplă:", height=75)
    if st.button("✨ Generează Univers cu AI"):
        if idee_scurta.strip():
            with st.spinner("AI-ul creează universul..."):
                univ = inventeaza_univers_capitol(model_ollama, idee_scurta)
                st.session_state["draft_nume"] = univ.get("chapter_name", "New Series")
                st.session_state["draft_tema"] = univ.get("expanded_premise", idee_scurta)
                st.session_state["draft_lume"] = univ.get("characters_and_world", "")
                st.rerun()

    nume_nou = st.text_input("Nume:", value=st.session_state.get("draft_nume", ""))
    tema_noua = st.text_area("Temă:", value=st.session_state.get("draft_tema", ""), height=80)
    lume_noua = st.text_area("Lume & Personaje:", value=st.session_state.get("draft_lume", ""), height=75)
    if st.button("💾 Salvează Capitol", type="primary"):
        if nume_nou.strip():
            new_id = str(uuid.uuid4())[:8]
            db["capitole"][new_id] = {
                "id": new_id, "nume": nume_nou.strip(), "descriere_tema": tema_noua.strip(),
                "personaje_si_lume": lume_noua.strip(), "paleta_culori": PALETE_DISPONIBILE[0],
                "creat_la": datetime.now().strftime("%Y-%m-%d %H:%M"), "episoade": []
            }
            db["capitol_activ"] = new_id
            salveaza_baza_capitole(db)
            st.rerun()

with tab_youtube:
    st.subheader("📤 Upload YouTube")
    ultimul_ep = capitol_curent["episoade"][-1] if capitol_curent["episoade"] else {}
    t_m = st.text_input("Titlu:", value=ultimul_ep.get("titlu", ""))
    d_m = st.text_area("Descriere:", value=ultimul_ep.get("descriere", ""), height=180)
    tags_m = st.text_input("Tag-uri:", value=", ".join(ultimul_ep.get("tags", ["animation"])))
    if st.button("🚀 Urcă pe YouTube", type="primary"):
        if os.path.exists("/root/rezultat.mp4") and os.path.exists("/root/token.json"):
            with st.spinner("Se încarcă..."):
                ryt = upload_youtube_seo("/root/rezultat.mp4", t_m, d_m, [x.strip() for x in tags_m.split(",")], "27", "public")
                st.success(f"Gata! Link: https://youtu.be/{ryt['id']}")