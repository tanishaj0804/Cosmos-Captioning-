import os
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

VIDEO_PATH        = "videos/traffic_crossing.mp4"
API_URL           = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL             = "Qwen/Qwen3-VL-8B-Instruct"
FRAMES_TO_EXTRACT = 8         
TARGET_WIDTH      = 480

SAFETY_PROMPT = (
    "These are frames extracted from a traffic video. "
    "Analyze this video strictly from a road safety perspective. "
    "Your response must cover the following points:\n\n"
    "1. WHAT IS HAPPENING: Briefly describe the scene.\n"
    "2. SAFETY HAZARDS: What specific dangers are visible? "
    "Look for jaywalking, distracted pedestrians, vehicles not stopping, "
    "lack of zebra crossing usage, overspeeding, poor visibility, or any other risk.\n"
    "3. WHO IS AT RISK: Which people or vehicles are in a dangerous situation and why?\n"
    "4. WHY THIS IS DANGEROUS: Explain clearly why the observed behavior could lead to accidents.\n"
    "5. RECOMMENDATIONS: What should pedestrians and drivers do differently to make this crossing safer?\n\n"
    "Be specific and detailed. Reference what you actually see in the frames."
)

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
        return 0.0

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
    
def analyze():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found at: {VIDEO_PATH}")
        return


    print(f"  Model   : {MODEL}")
    print(f"  Video   : {VIDEO_PATH}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    duration = get_duration(VIDEO_PATH)
    print(f"Duration       : {duration}s")
    print(f"Extracting {FRAMES_TO_EXTRACT} frames...")

    tmp_dir, frame_paths = extract_frames(VIDEO_PATH, FRAMES_TO_EXTRACT, duration)

    if not frame_paths:
        print("No frames extracted.")
        return

    print(f"{len(frame_paths)} frames extracted")

    content = []
    for frame_path in frame_paths:
        img_b64 = encode_image(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    content.append({"type": "text", "text": SAFETY_PROMPT})

    shutil.rmtree(tmp_dir)

    payload = {
        "model": MODEL,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"Sending to DeepInfra...")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        print("  SAFETY ANALYSIS OUTPUT:\n")
        print(result)

        os.makedirs("output", exist_ok=True)
        output_path = "output/traffic_safety_analysis.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("TRAFFIC SAFETY ANALYSIS\n")
            f.write(f"Video   : {VIDEO_PATH}\n")
            f.write(f"Model   : {MODEL}\n")
            f.write(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(result)
        print(f"\nSaved to: {output_path}")
    else:
        print(f"  ERROR {response.status_code}: {response.text}")


if __name__ == "__main__":
    analyze()