import streamlit as st
from PIL import Image, ImageOps
import os
import pandas as pd
from main import Indexer
from math import floor

# Init search engine
@st.cache_resource
def load_search():
    return Indexer()

@st.cache_data
def search(query, limit, filters):
    return search_engine.semantic_search(query, limit=limit, tag_filters=filters)

@st.cache_data
def load_image(path):
    with Image.open(path) as img:
        return img.copy()

search_engine = load_search()

# UI
st.title("NAS Video Search")
query = st.text_input("Search", placeholder="Enter a query...")

col1, col2 = st.columns(2)

with col1:
    limit = st.slider("Results", 1, 50, 10)

with col2:
    filters = st.multiselect("Tags", search_engine.get_tags())

# Video player
@st.dialog("Preview", width="large")
def video_player():
    if "selected_video" in st.session_state:
        video_path = st.session_state["selected_video"]
        thumb_path = st.session_state["thumbnail_path"]
        start_t = int(st.session_state["start_time"])
        end_t = int(st.session_state["end_time"])
        tags = st.session_state["tags"]

        if video_path and os.path.exists(video_path):
            video_placeholder = st.empty()
            video_placeholder.empty()
            video_placeholder.video(video_path, start_time=start_t)

            data = pd.DataFrame([{
                "Video": video_path[:-4],
                "Clip": os.path.basename(thumb_path)[:-4],
                "Start": f"{floor(start_t/60)}m {floor(start_t%60)}s",
                "End": f"{floor(end_t/60)}m {floor(end_t%60)}s",
                "Tags": '*None*' if not tags else ' | '.join(tags)
            }])

            st.table(data)
        else:
            st.error("Video not found")

@st.dialog("Tag Clips", width="large")
def tag_videos(untag=False):
    cola, colb = st.columns([0.8, 0.2])
    
    with cola:
        tags = st.multiselect("Enter tag", options=search_engine.get_tags(), accept_new_options=True)
    
    with colb:
        st.space("small")
        button = st.button(f"{'T' if not untag else 'Unt'}ag Selected Items", type="primary")

    if button and tags != []:
        if not untag:
            search_engine.tag_clips([r["thumbnail_path"] for r in st.session_state.selected], [tag.lower().replace("\"", "").replace("'", "").replace(" ", "-") for tag in tags])
        else:
            search_engine.untag_clips([r["thumbnail_path"] for r in st.session_state.selected], [tag.lower().replace("\"", "").replace("'", "").replace(" ", "-") for tag in tags])
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    for i, s in enumerate(st.session_state.selected):
        col1, col2 = st.columns([0.2, 0.8])

        with col1:
            try:
                img = load_image(s["thumbnail_path"])
                # enforce fixed height while preserving aspect ratio
                fixed_height = 160
                img = ImageOps.fit(
                    img,
                    (300, fixed_height),  # width, height
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5)
                )

                st.image(img, width="stretch")
            except Exception as e:
                st.error(f"Image error: {e}")
                continue

        with col2:
            st.caption(f"""
                **Score:** {(s['score']*100):.2f}
                **Timestamp:** {floor(s['start_time']/60)}m {floor(s['start_time']%60)}s - {floor(s['end_time']/60)}m {floor(s['end_time']%60)}s
            """)
            st.caption(f"**Scene title:** {os.path.basename(s['thumbnail_path'])}")
            st.caption(f"**Tags:** {'*None*' if not s['tags'] else ' | '.join(s['tags'])}")

# Run search
if query:
    results = search(query, limit=limit, filters=filters)

    if not results:
        st.warning("No results found")
    else:
        if "selected" not in st.session_state:
            st.session_state.selected = []

        if "checkboxes" not in st.session_state:
            st.session_state.checkboxes = []
        else:
            st.session_state.checkboxes.clear()

        thumb_paths = [r["thumbnail_path"] for r in results]
        st.session_state.selected = [s for s in st.session_state.selected if s["thumbnail_path"] in thumb_paths]
        st.write(f"Found {len(results)} results")
                
        def toggle_all():
            for key in st.session_state.checkboxes:
                st.session_state[key] = st.session_state.check_all

        col1, col2, col3 = st.columns([3.5, 1, 1.2])

        with col1:
            check_all = st.checkbox("Select All", key="check_all", on_change=toggle_all)

        with col2:
            tag_clips = st.button(f"Tag selected", key="tag_button", type="primary")
            if tag_clips and len(st.session_state.selected) > 0:
                tag_videos()

        with col3:
            untag_clips = st.button(f"Untag selected", key="untag_button", type="primary")
            if untag_clips and len(st.session_state.selected) > 0:
                tag_videos(untag=True)

        # Display grid
        for i, r in enumerate(results):
            cols = st.columns([0.2, 1, 3, 0.7])

            thumb_path = r["thumbnail_path"]
            video_path = r["file_path"]
            start_time = r["start_time"]
            end_time = r["end_time"]
            transcript = r["text_transcript"]
            tags = r["tags"]

            with cols[0]:
                key = f"check_{r['file_hash']}_{r['scene_index']}"

                checkbox = st.checkbox(" ", key=key)
                st.session_state.checkboxes.append(key)

                selected_paths = {s["thumbnail_path"] for s in st.session_state.selected}

                if checkbox:
                    if r["thumbnail_path"] not in selected_paths:
                        st.session_state.selected.append(r)
                else:
                    st.session_state.selected = [
                        s for s in st.session_state.selected
                        if s["thumbnail_path"] != r["thumbnail_path"]
                    ]

            with cols[1]:
                try:
                    img = load_image(thumb_path)
                    # enforce fixed height while preserving aspect ratio
                    fixed_height = 160
                    img = ImageOps.fit(
                        img,
                        (300, fixed_height),  # width, height
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5)
                    )

                    st.image(img, width="stretch")
                except Exception as e:
                    st.error(f"Image error: {e}")
                    continue

            with cols[2]:
                st.caption(f"""
                    **Score:** {(r['score']*100):.2f}
                    **Timestamp:** {floor(start_time/60)}m {floor(start_time%60)}s - {floor(end_time/60)}m {floor(end_time%60)}s
                """)
                st.caption(f"**Scene title:** {os.path.basename(thumb_path)}")

            with cols[3]:
                if st.button("Open", key=f"btn_{i}"):
                    st.session_state["selected_video"] = video_path
                    st.session_state["start_time"] = start_time
                    st.session_state["end_time"] = end_time
                    st.session_state["thumbnail_path"] = thumb_path
                    st.session_state["tags"] = tags
                    video_player()