# Requirements
- Install Python3.11: https://www.python.org/downloads/release/python-3110/

# How to
*First time*
Open terminal. 
Clone repo: `git clone https://github.com/dylanwynne-tenicor/video-embeddings-media.git && cd video-embeddings-media`
Setup venv: `python3.11 -m venv .venv && source .venv/bin/activate`
Install requirements: `pip install -r requirements.txt`
Initialize workspace: `python main.py && mkdir input_videos`
Add videos to the input_videos folder now.

*After setup*
To index all videos in the input_videos folder, run: `python main.py`
(This will skip any videos that have already been indexed)

To run the frontend app, run: `streamlit run app.py --server.headless true`
(The url printed after Network URL is accessible to anyone on the same WiFi network)