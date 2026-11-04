"""
app/app.py
-----------
PotholeIQ — Minimal Streamlit UI.

Run:  streamlit run app/app.py

Tabs:
  1. Classifier   — pothole / normal + confidence
  2. Segmentation — mask overlay + severity level
  3. Grad-CAM     — heatmap on the input image
  4. Anomaly      — autoencoder reconstruction error

Note: model loading stubs will be replaced as each module is completed.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="PotholeIQ", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🕳️ PotholeIQ — Road Damage Analysis")
st.caption("Upload a road image to run the full detection pipeline.")

uploaded = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("👆 Upload a road image to begin.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

col1, col2 = st.columns([1, 2])
with col1:
    st.image(img, caption="Input image", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_cls, tab_seg, tab_xai, tab_ae = st.tabs(
    ["🔍 Classifier", "🗺️ Segmentation", "🔥 Grad-CAM", "🤖 Anomaly"]
)

with tab_cls:
    st.subheader("Classification")
    st.warning("⏳ Classifier not yet trained (M3/M4 pending). "
               "This tab will show label + confidence once weights are available.")

with tab_seg:
    st.subheader("Segmentation & Severity")
    st.warning("⏳ U-Net not yet trained (M5 pending).")

with tab_xai:
    st.subheader("Grad-CAM Explanation")
    st.warning("⏳ Grad-CAM not yet implemented (M6 pending).")

with tab_ae:
    st.subheader("Anomaly Score")
    st.warning("⏳ Autoencoder not yet trained (M8 pending).")

st.divider()
st.caption(
    "PotholeIQ — B.Tech Deep Learning Course Project. "
    "Results are course-level; not claiming state-of-the-art."
)
