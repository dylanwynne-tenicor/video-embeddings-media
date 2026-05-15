if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Extremely important to do this first
    import subprocess, tempfile
    import hashlib
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    from scenedetect import detect, AdaptiveDetector
    from faster_whisper import WhisperModel
    from concurrent.futures import ThreadPoolExecutor, as_completed

import os
from collections import defaultdict
from dataclasses import dataclass
import lancedb
import pyarrow as pa
import torch
import open_clip

@dataclass
class SceneMetadata:
    file_path: str
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    thumbnail_path: str
    transcript: str = ""
    file_hash: str = ""

class Indexer:
    def __init__(
        self,
        # video_root_dir = "/Volumes/Media/Video/",
        video_root_dir = "input_videos",
        output_dir = "resources",
        db_path = "db",
        whisper_model_name = "base",
        clip_model_name = "ViT-B-32",
        clip_preprocess_name = "openai",
        device = "cpu"
    ):
        self.video_root_dir = video_root_dir
        self.output_dir = output_dir
        self.db_path = db_path

        # Setup Directories
        self.thumbnails_dir = os.path.join(output_dir, "thumbnails")
        os.makedirs(self.thumbnails_dir, exist_ok=True)

        self.device = device

        # Setup OpenCLIP
        self.clip_model_name = clip_model_name
        self.clip_preprocess_name = clip_preprocess_name
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

        # Setup faster-whisper
        self._whisper_model = None
        self.whisper_model_name = whisper_model_name

        # Setup db
        self.db = lancedb.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        video_metadata_schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 512)),  # CLIP embedding
            pa.field("text_transcript", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("start_time", pa.float64()),
            pa.field("end_time", pa.float64()),
            pa.field("thumbnail_path", pa.string()),
            pa.field("scene_index", pa.int32()),
            pa.field("file_hash", pa.string()),
            pa.field("duration", pa.float64()),
            pa.field("tags", pa.list_(pa.string()))
        ])

        if "scenes" not in self.db.table_names():
            self.table = self.db.create_table("scenes", schema=video_metadata_schema)
            self.table.create_fts_index("text_transcript")
        else:
            self.table = self.db.open_table("scenes")

        # self.table.create_fts_index("text_transcript")

        tags_schema = pa.schema([
            pa.field("tag", pa.string())
        ])

        if "tags" not in self.db.table_names():
            self.tags_table = self.db.create_table("tags", schema=tags_schema)
        else:
            self.tags_table = self.db.open_table("tags")
        
    @property
    def whisper_model(self):
        if self._whisper_model is None:
            # Consider changing these params: device (for gpu),  compute type (float16 or in8)
            self._whisper_model = WhisperModel(self.whisper_model_name)
        return self._whisper_model

    @property
    def clip_model(self):
        if self._clip_model is None:
            self._init_clip()
        return self._clip_model

    @property
    def clip_preprocess(self):
        if self._clip_preprocess is None:
            self._init_clip()
        return self._clip_preprocess

    @property
    def clip_tokenizer(self):
        if self._clip_tokenizer is None:
            self._init_clip()
        return self._clip_tokenizer

    def _init_clip(self):
        # Consider changing device param
        model, preprocess, _ = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_preprocess_name
        )
        self._clip_model = model.to(self.device).eval()
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def hash_file(self, video_path):
        hash_md5 = hashlib.md5()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_transcript(self, video_path, start_time, end_time):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            duration = end_time - start_time
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", video_path,
                "-ac", "1",
                "-ar", "16000",
                "-y",
                temp_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            segments, _ = self.whisper_model.transcribe(temp_path, beam_size=5)
            transcript = " ".join([segment.text.strip() for segment in segments])

            if os.path.exists(temp_path):
                os.unlink(temp_path)

            return transcript

    def get_scenes(self, video_path):
        if not os.path.isfile(video_path):
            return []

        scene_list = detect(video_path, AdaptiveDetector())
        scenes = []
        for i, scene in enumerate(scene_list):
            scenes.append(SceneMetadata(
                file_path=video_path,
                scene_index=i,
                start_time=scene[0].get_seconds(),
                end_time=scene[1].get_seconds(),
                duration=scene[1].get_seconds() - scene[0].get_seconds(),
                thumbnail_path="",
                file_hash=self.hash_file(video_path)
            ))

        return scenes

    def get_thumbnail(self, video_path, scene):
        middle_time = scene.start_time + (scene.duration / 2)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        thumbnail_name = f"{video_name}_scene_{scene.scene_index:04d}.jpg"
        thumbnail_path = os.path.join(self.thumbnails_dir, thumbnail_name)
        
        cmd = [
            "ffmpeg",
            "-ss", str(middle_time),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            thumbnail_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            scene.thumbnail_path = thumbnail_path
            return thumbnail_path
        except subprocess.CalledProcessError as e:
            print(f"Error extracting thumbnail: {e}")
            return ""

    def get_image_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.clip_model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
                
            return features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return np.zeros(self.clip_model.visual.output_dim)

    def process_scene(self, video_path, scene, file_hash):
        try:
            thumbnail_path = self.get_thumbnail(video_path, scene)
            if not thumbnail_path:
                return None
            
            transcript = self.get_transcript(video_path, scene.start_time, scene.end_time)
            embedding = self.get_image_embedding(thumbnail_path)
            
            return {
                "vector": embedding.tolist(),
                "text_transcript": transcript,
                "file_path": video_path,
                "start_time": float(scene.start_time),
                "end_time": float(scene.end_time),
                "thumbnail_path": thumbnail_path,
                "scene_index": scene.scene_index,
                "file_hash": file_hash,
                "duration": float(scene.duration)
            }
        except Exception as e:
            print(f"Error processing scene {scene.scene_index}: {e}")
            return None

    def process_video(self, video_path, force_reindex = False):
        # Check if already processed
        file_hash = self.hash_file(video_path)
        if not force_reindex:
            existing = self.table.search().where(f"file_hash = '{file_hash}'").to_pandas()
            if not existing.empty:
                print(f"Skipping {os.path.basename(video_path)} - already indexed")
                return 0
        
        scenes = self.get_scenes(video_path)
        if not scenes:
            print(f"No scenes detected in {os.path.basename(video_path)}")
            return 0
        
        print(f"Found {len(scenes)} scenes in {os.path.basename(video_path)}")
        
        # Process scenes in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for scene in scenes:
                future = executor.submit(self.process_scene, video_path, scene, file_hash)
                futures.append(future)
            
            scene_data = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
                try:
                    result = future.result()
                    if result:
                        scene_data.append(result)
                except Exception as e:
                    print(f"Error processing scene: {e}")
        
        if scene_data:
            self.table.add(scene_data)
            print(f"Indexed {len(scene_data)} scenes from {os.path.basename(video_path)}")
            
        return len(scene_data)

    def process_video_dir(self):
        for root, _, files in os.walk(self.video_root_dir):
            for file in files:
                if file[-4:].lower() != ".mp4":
                    continue

                video_path = os.path.join(root, file)
                print(f"Beginning indexing on: {video_path}")
                self.process_video(video_path)
                print(f"Completed indexing on: {video_path}")

    def tag_clip(self, thumbnail_path, tag):
        tag_exists = self.tags_table.search().where(f"tag = '{tag}'").to_pandas()
        if tag_exists.empty:
            self.tags_table.add([{ "tag": tag }])

        scene = self.table.search().where(f"thumbnail_path = '{thumbnail_path}'").to_pandas()
        if scene.empty:
            return "ERROR: Scene not found."

        if scene.iloc[0]["tags"] is None:
            tags = [tag]
        else:
            tags = list(scene.iloc[0]["tags"])

        if tag not in tags:
            tags.append(tag)
            quoted = ",".join(f"'{t}'" for t in tags)
            self.table.update(where=f"thumbnail_path = '{thumbnail_path}'", values_sql={ "tags": f"make_array({quoted})" })

    def tag_clips(self, thumbnail_path_list, tags):
        for tag in tags:
            for thumbnail_path in thumbnail_path_list:
                self.tag_clip(thumbnail_path, tag)

    def get_tags(self):
        results = self.tags_table.search().to_pandas()
        return results["tag"].to_list()

    def untag_clip(self, thumbnail_path, tag):
        scene = self.table.search().where(f"thumbnail_path = '{thumbnail_path}'").to_pandas()
        if scene.empty:
            return "ERROR: Scene not found."

        if scene.iloc[0]["tags"] is None:
            return "ERROR: Tag not found on scene."

        if tag not in scene.iloc[0]["tags"]:
            return "ERROR: Tag not found on scene."

        tags = list(scene.iloc[0]["tags"])
        tags.remove(tag)

        # NOTE: This is different than taggin because untag must be able to push an empty list to a row
        quoted = ",".join(f"'{t}'" for t in tags)
        self.table.update(where=f"thumbnail_path = '{thumbnail_path}'", values_sql={ "tags": f"make_array({quoted})" })

        self.purge_tags([tag])

    def untag_clips(self, thumbnail_path_list, tags):
        for tag in tags:
            for thumbnail_path in thumbnail_path_list:
                self.untag_clip(thumbnail_path, tag)

    def purge_tags(self, tags=None):
        if tags is None:
            tags = self.get_tags()

            if tags is None:
                return
        
            for tag in tags:
                results = self.table.search().where(f"array_has(tags, '{tag}')").limit(1).to_list()
                if len(results) == 1:
                    continue
                self.tags_table.delete(f"tag = '{tag}'")
        else:
            if tags is None or tags == []:
                return
        
            for tag in tags:
                results = self.table.search().where(f"array_has(tags, '{tag}')").limit(1).to_list()
                if len(results) == 1:
                    continue
                self.tags_table.delete(f"tag = '{tag}'")

    def semantic_search(self, query, limit = 20, k = 60, tag_filters = []):
        # Embed query
        with torch.no_grad():
            text_tokens = self.clip_tokenizer([query]).to(self.device)
            query_embedding = self.clip_model.encode_text(text_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            query_vector = query_embedding.cpu().numpy()[0].tolist()

        # Collect excess results
        fetch_k = max(limit * 10, 100)

        tag_conditions = " OR ".join([f"array_has(tags, '{tag}')" for tag in tag_filters])

        vector_results = (
            self.table.search(query_vector, query_type="vector")
            .where(tag_conditions)
            .limit(fetch_k)
            .to_list()
        )
        # DEBUG: Remove
        # print("Vector results:", len(vector_results))

        text_results = (
            self.table.search(query, query_type="fts")
            .where(tag_conditions)
            .limit(fetch_k)
            .to_list()
        )
        # DEBUG: Remove
        # print("Text results:", len(text_results))

        scores = defaultdict(lambda: {"rrf": 0.0, "row": None})

        def add_rrf(results, key="vector"):
            for rank, r in enumerate(results):
                file_id = r["thumbnail_path"]
                scores[file_id]["rrf"] += 1.0 / (k + rank + 1)
                scores[file_id]["row"] = r

        add_rrf(vector_results, "vector")
        add_rrf(text_results, "text")

        final = []
        for file_id, s in scores.items():
            if s["row"] is None:
                continue

            r = s["row"].copy()
            r["score"] = s["rrf"]
            final.append(r)

        final.sort(key=lambda x: x["score"], reverse=True)
        return final[:limit]
    
    def search_by_tag(self, tag, limit = 20): # TODO
        pass
        
def main():
    indexer = Indexer("input_videos")
    indexer.process_video_dir()

if __name__ == "__main__":
    main()