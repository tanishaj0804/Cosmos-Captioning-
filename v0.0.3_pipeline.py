
import os
import base64
import requests
import csv
import subprocess
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DEEPINFRA_API_KEY")

if not API_KEY:
    raise EnvironmentError("DEEPINFRA_API_KEY not found. Please check your .env file.")

VIDEOS_FOLDER   = "videos"
OUTPUT_FOLDER   = "output"
OUTPUT_FILE     = os.path.join(OUTPUT_FOLDER, "results.csv")

API_URL         = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL           = "Qwen/Qwen3-VL-8B-Instruct"  

FRAMES_TO_EXTRACT = 5       
TARGET_WIDTH      = 480    
PROMPT            = "These are frames extracted from a video. Describe what is happening in this video in detail. Focus on objects, people, actions, and the environment."


def extract_frames(video_path, num_frames=5):
    tmp_dir = tempfile.mkdtemp()

    # Get video duration first
    duration_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f" Video duration : {duration:.1f}s")

    # Extract frames at evenly spaced intervals
    interval = duration / (num_frames + 1)
    frame_paths = []

    for i in range(1, num_frames + 1):
        timestamp = interval * i
        frame_path = os.path.join(tmp_dir, f"frame_{i:02d}.jpg")

        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={TARGET_WIDTH}:-2",
            "-q:v", "3",
            "-y",
            frame_path
        ]
        subprocess.run(cmd, capture_output=True)

        if os.path.exists(frame_path):
            frame_paths.append(frame_path)

    print(f"  → Extracted {len(frame_paths)} frames")
    return tmp_dir, frame_paths

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_video_description(video_path):
    print(f"Extracting frames...")
    tmp_dir, frame_paths = extract_frames(video_path, FRAMES_TO_EXTRACT)

    if not frame_paths:
        return "ERROR: No frames could be extracted from video"

    content = []
    for frame_path in frame_paths:
        img_b64 = encode_image(frame_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        })

    content.append({
        "type": "text",
        "text": PROMPT
    })

    shutil.rmtree(tmp_dir)

    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"Sending {FRAMES_TO_EXTRACT} frames to {MODEL}...")
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"ERROR {response.status_code}: {response.text}"

def run_pipeline():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(".mp4")]

    if not video_files:
        print("No MP4 files found in the 'videos/' folder. Please add some and retry.")
        return

    print(f"  Cosmos Captioning Pipeline  |  v0.0.3")
    print(f"  Host     : DeepInfra")
    print(f"  Model    : {MODEL}")
    print(f"  Approach : Frame extraction ({FRAMES_TO_EXTRACT} frames per video)")
    print(f"  Videos   : {len(video_files)}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    results = []

    for idx, filename in enumerate(video_files, start=1):
        video_path = os.path.join(VIDEOS_FOLDER, filename)
        print(f"[{idx}/{len(video_files)}] Processing: {filename}")

        description = get_video_description(video_path)

        results.append({
            "video_file"  : filename,
            "description" : description,
            "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"Done\n")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["video_file", "description", "timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Results saved to : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()