import os
import json
import base64
import requests
import subprocess
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not API_KEY:
    raise EnvironmentError("DEEPINFRA_API_KEY not found. Please check your .env file.")

VIDEOS_FOLDER     = "videos"
OUTPUT_FOLDER     = "output"
OUTPUT_JSON       = os.path.join(OUTPUT_FOLDER, "results.json")

API_URL           = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL             = "Qwen/Qwen3-VL-8B-Instruct"
FRAMES_TO_EXTRACT = 5
TARGET_WIDTH      = 480

SCENE_DETECT_PROMPT = """
Look at these video frames and identify the scene type.
Reply with ONLY a JSON object in this exact format, nothing else:
{
  "scene_type": "<one of: traffic, robotics, factory, sports, crowd, indoor, nature, other>",
  "scene_description": "<one sentence describing what you see>",
  "key_subjects": ["<subject1>", "<subject2>"]
}
"""

SCENE_PROMPTS = {

    "traffic": """
Analyze these traffic video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "traffic",
  "timestamp": "<current datetime>",
  "location_description": "<describe the road/junction>",
  "vehicles": {
    "vehicle_1": {
      "type": "<car/truck/bike/bus>",
      "colour": "<colour or unknown>",
      "number_plate": "<plate or not visible>",
      "direction": "<direction of travel>",
      "speed_estimate": "<slow/moderate/fast>",
      "dangerous": "<yes/no>",
      "reason": "<why dangerous or safe>"
    }
  },
  "pedestrians": {
    "pedestrian_1": {
      "action": "<what they are doing>",
      "at_risk": "<yes/no>",
      "reason": "<why at risk or not>"
    }
  },
  "safety_summary": {
    "overall_risk_level": "<low/medium/high>",
    "violations_detected": ["<violation1>", "<violation2>"],
    "recommendations": ["<recommendation1>", "<recommendation2>"]
  }
}
Add more vehicles/pedestrians as needed based on what you see.
""",

    "robotics": """
Analyze these robotics video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "robotics",
  "timestamp": "<current datetime>",
  "environment": "<describe the setting>",
  "robots": {
    "robot_1": {
      "type": "<arm/humanoid/mobile/other>",
      "current_action": "<what it is doing>",
      "component_handled": "<object or part being handled>",
      "precision": "<high/medium/low>",
      "anomaly_detected": "<yes/no>",
      "anomaly_detail": "<describe anomaly or none>"
    }
  },
  "assembly_steps_observed": ["<step1>", "<step2>"],
  "safety_summary": {
    "overall_risk_level": "<low/medium/high>",
    "anomalies": ["<anomaly1>"],
    "recommendations": ["<recommendation1>"]
  }
}
Add more robots as needed.
""",

    "factory": """
Analyze these factory video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "factory",
  "timestamp": "<current datetime>",
  "environment": "<describe the factory area>",
  "machines": {
    "machine_1": {
      "type": "<conveyor/press/drill/other>",
      "status": "<running/idle/error>",
      "anomaly_detected": "<yes/no>",
      "anomaly_detail": "<describe or none>"
    }
  },
  "workers": {
    "worker_1": {
      "action": "<what they are doing>",
      "ppe_compliant": "<yes/no/unknown>",
      "at_risk": "<yes/no>",
      "reason": "<why or not>"
    }
  },
  "safety_summary": {
    "overall_risk_level": "<low/medium/high>",
    "violations_detected": ["<violation1>"],
    "recommendations": ["<recommendation1>"]
  }
}
""",

    "sports": """
Analyze these sports video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "sports",
  "timestamp": "<current datetime>",
  "sport_type": "<football/basketball/cricket/other>",
  "environment": "<describe the venue>",
  "players": {
    "player_1": {
      "action": "<what they are doing>",
      "team": "<team colour or unknown>",
      "notable_event": "<goal/foul/tackle/none>"
    }
  },
  "game_summary": {
    "key_events": ["<event1>", "<event2>"],
    "overall_activity_level": "<low/medium/high>"
  }
}
""",

    "crowd": """
Analyze these crowd video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "crowd",
  "timestamp": "<current datetime>",
  "location_description": "<describe the location>",
  "crowd_analysis": {
    "estimated_count": "<number or range>",
    "density": "<sparse/moderate/dense>",
    "movement_direction": "<describe flow>",
    "mood": "<calm/excited/panic/unknown>"
  },
  "safety_summary": {
    "overall_risk_level": "<low/medium/high>",
    "concerns": ["<concern1>", "<concern2>"],
    "recommendations": ["<recommendation1>"]
  }
}
""",

    "other": """
Analyze these video frames. Reply ONLY with a valid JSON object:
{
  "scene_type": "<describe the actual scene>",
  "timestamp": "<current datetime>",
  "environment": "<describe the setting>",
  "subjects": {
    "subject_1": {
      "type": "<what it is>",
      "action": "<what it is doing>",
      "notable_detail": "<any important observation>"
    }
  },
  "summary": {
    "main_activity": "<describe overall activity>",
    "key_observations": ["<observation1>", "<observation2>"],
    "anomalies": ["<anomaly or none>"]
  }
}
Add more subjects as needed.
"""
}

SCENE_PROMPTS["indoor"]  = SCENE_PROMPTS["other"]
SCENE_PROMPTS["nature"]  = SCENE_PROMPTS["other"]

def get_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return round(float(result.stdout.strip()), 1)
    except:
        return 30.0

def extract_frames(video_path, num_frames, duration):
    tmp_dir  = tempfile.mkdtemp()
    interval = duration / (num_frames + 1)
    frame_paths = []

    for i in range(1, num_frames + 1):
        timestamp  = interval * i
        frame_path = os.path.join(tmp_dir, f"frame_{i:02d}.jpg")
        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={TARGET_WIDTH}:-2",
            "-q:v", "3", "-y", frame_path
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)

    return tmp_dir, frame_paths

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_image_content(frame_paths):
    content = []
    for frame_path in frame_paths:
        img_b64 = encode_image(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    return content

def call_api(content):
    payload = {
        "model": MODEL,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")

def parse_json_response(text):
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except:
        return {"raw_response": text, "parse_error": "Could not parse JSON"}

def process_video(video_path, filename):
    print(f"\nExtracting frames...")
    duration = get_duration(video_path)
    tmp_dir, frame_paths = extract_frames(video_path, FRAMES_TO_EXTRACT, duration)

    if not frame_paths:
        shutil.rmtree(tmp_dir)
        return {"error": "No frames extracted", "video_file": filename}

    image_content = build_image_content(frame_paths)
    shutil.rmtree(tmp_dir)

    print(f"  → Step 1: Detecting scene type...")
    scene_content = image_content + [{"type": "text", "text": SCENE_DETECT_PROMPT}]
    scene_raw     = call_api(scene_content)
    scene_info    = parse_json_response(scene_raw)
    scene_type    = scene_info.get("scene_type", "other").lower()
    print(f"  → Scene detected : {scene_type}")

    print(f"  → Step 2: Generating structured JSON output...")
    json_prompt   = SCENE_PROMPTS.get(scene_type, SCENE_PROMPTS["other"])
    json_content  = image_content + [{"type": "text", "text": json_prompt}]
    json_raw      = call_api(json_content)
    json_output   = parse_json_response(json_raw)

    result = {
        "video_file"      : filename,
        "processed_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": duration,
        "frames_extracted": len(frame_paths),
        "model_used"      : MODEL,
        "scene_detection" : scene_info,
        "analysis"        : json_output
    }

    return result

def run_pipeline():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    video_files = sorted([f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(".mp4")])

    if not video_files:
        print("No MP4 files found in 'videos/' folder.")
        return

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  Host     : DeepInfra")
    print(f"  Model    : {MODEL}")
    print(f"  Approach : Scene detection + Adaptive JSON output")
    print(f"  Videos   : {len(video_files)}")
    print(f"  Started  : {run_time}")


    all_results = []

    for idx, filename in enumerate(video_files, start=1):
        video_path = os.path.join(VIDEOS_FOLDER, filename)
        print(f"\n[{idx}/{len(video_files)}] Processing: {filename}")

        try:
            result = process_video(video_path, filename)
            all_results.append(result)

            safe_name    = os.path.splitext(filename)[0]
            per_video_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}_analysis.json")
            with open(per_video_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Done → {per_video_path}")

        except Exception as e:
            print(f"Error: {e}")
            all_results.append({"video_file": filename, "error": str(e)})

    master = {
        "pipeline_version" : "v0.1.0",
        "run_at"           : run_time,
        "total_videos"     : len(video_files),
        "results"          : all_results
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print(f"  Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()