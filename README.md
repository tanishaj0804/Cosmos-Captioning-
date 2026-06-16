# Cosmos Captioning Pipeline

A PoC pipeline that takes video files as input and generates detailed text descriptions of what's happening in each video.


### Approach

Since direct video input is not supported by available APIs at this stage, the pipeline works by:

Extracting 5 evenly spaced frames from each video using ffmpeg
Sending those frames as images to a Vision Language Model (Qwen3-VL) via DeepInfra
Receiving a detailed text description of the video content
Saving all results to a CSV file

### Setup
1. Clone this repo
2. Create a virtual environment and activate it

         python -m venv venv
             venv\Scripts\activate
4. Install dependencies
5. Create a .env file and add your DeepInfra API key
   
            DEEPINFRA_API_KEY=your_key_here
7. Drop your .mp4 files into the videos/ folder
8. Run

           python v0.0.3_pipeline.py

Results are saved to output/results.csv with columns: video_file, description, timestamp


### Current Limitations

NVIDIA Cosmos3-Nano Reasoner cloud API is not yet publicly live (returns 404) — known issue on NVIDIA developer forums
DeepInfra hosts Cosmos3-Nano as a video generator, not a reasoner — not suitable for video→text
Current workaround uses frame extraction + Qwen3-VL which produces high quality descriptions
Will switch to Cosmos Reasoner NIM once NVIDIA endpoint goes live or GPU access is available
