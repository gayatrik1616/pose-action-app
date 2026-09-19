"""
app.py
------
Streamlit web app to demo the pipeline visually — this is the easiest way
to "see the model running" without touching the command line.

Run locally:
    streamlit run app.py

Then open the URL it prints (usually http://localhost:8501) in your browser.
"""

import time
import tempfile
import os

import cv2
import streamlit as st

from src.pipeline import CrowdPoseActionPipeline

st.set_page_config(page_title="Crowd Pose & Action", layout="wide", page_icon="🎯")

# Inject Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit default UI components (footer only) */
    footer {visibility: hidden;}
    
    /* Center the title */
    .title-text {
        text-align: center;
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
        font-size: 3em;
        font-weight: 800;
    }
    
    .caption-text {
        text-align: center;
        font-size: 1.2em;
        color: #A0AEC0;
        margin-bottom: 40px;
    }
    
    /* Styling for Streamlit metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>Human Pose Estimation & Action Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption-text'>YOLOv8-Pose (multi-person) + built-in tracking + per-person action classification</p>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_pipeline(config_path="config.yaml"):
    return CrowdPoseActionPipeline(config_path=config_path)


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    with st.expander("🎥 Input Source", expanded=True):
        source_type = st.radio("Select source format", ["Upload video", "Webcam (snapshot)"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🧠 Model Ecosystem")
    st.success("**Pose Model:** YOLOv8-Pose")
    st.success("**Tracker:** ByteTrack")
    st.info("**Action Recognizer:** Rule-based / LSTM")

pipeline = get_pipeline()

if source_type == "Upload video":
    st.markdown("### 📤 Upload Media")
    uploaded = st.file_uploader("Upload a video (mp4/avi/mov)", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name

        st.video(video_path)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_btn = st.button("▶ Run Pose Estimation + Action Recognition", use_container_width=True, type="primary")

        if run_btn:
            cap = cv2.VideoCapture(video_path)
            
            st.markdown("---")
            st.markdown("### ⚙️ Processing Video...")
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0, text="Processing frames...")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            action_tally = {}
            frame_idx = 0
            t0 = time.time()

            out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
            writer = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, results = pipeline.process_frame(frame)

                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
                writer.write(annotated)

                for r in results:
                    action_tally[r["action"]] = action_tally.get(r["action"], 0) + 1

                frame_idx += 1
                if frame_idx % 3 == 0:  # update UI every few frames for speed
                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        caption=f"Frame {frame_idx}/{total_frames}",
                    )
                    with stats_placeholder.container():
                        st.markdown("### 📊 Real-time Statistics")
                        cols = st.columns(4)
                        cols[0].metric("👥 People Tracked", len(results))
                        
                        # Display top actions so far
                        sorted_actions = sorted(action_tally.items(), key=lambda x: x[1], reverse=True)
                        for i in range(min(3, len(sorted_actions))):
                            action_name, count = sorted_actions[i]
                            cols[i+1].metric(f"🎯 {action_name}", count)
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            if writer is not None:
                writer.release()

            elapsed = time.time() - t0
            progress_bar.empty()
            st.success(f"✨ Done! Processed {frame_idx} frames in {elapsed:.1f}s "
                       f"({frame_idx / max(elapsed,1e-6):.2f} FPS).")
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("📈 Final Action Tally")
                st.bar_chart(action_tally)

            with res_col2:
                st.subheader("📥 Download Results")
                st.info("Your video has been successfully annotated with bounding boxes, keypoints, and action classes.")
                with open(out_path, "rb") as f:
                    st.download_button("⬇ Download Annotated Video", f,
                                        file_name="annotated_output.mp4", mime="video/mp4", 
                                        use_container_width=True, type="primary")

else:
    st.info("Take a snapshot from your webcam to test pose estimation on a single frame.")
    img_file = st.camera_input("Capture a frame")
    if img_file is not None:
        import numpy as np
        from PIL import Image

        image = Image.open(img_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated, results = pipeline.process_frame(frame)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Result")
        st.write(f"People detected: {len(results)}")
        for r in results:
            st.write(f"ID {r['track_id']}: **{r['action']}** (conf={r['conf']:.2f})")
