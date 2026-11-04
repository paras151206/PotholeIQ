"""
app/app.py
-----------
PotholeIQ — Streamlit UI (Module M10)

Run:
    streamlit run app/app.py

Tabs:
  1. Classifier   — pothole/normal + confidence bar
  2. Segmentation — mask overlay + severity badge
  3. Grad-CAM     — heatmap overlay explaining the classifier's decision
  4. Anomaly      — autoencoder reconstruction + anomaly score gauge
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

import streamlit as st

from src.utils.config import CFG
from src.utils.seed import set_all_seeds
from src.data.dataset import get_transforms

set_all_seeds(CFG.seed)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = ROOT / CFG.paths.checkpoints_dir
H, W       = CFG.data.image_size
MEAN = torch.tensor(CFG.augmentation.normalize_mean).view(3, 1, 1)
STD  = torch.tensor(CFG.augmentation.normalize_std).view(3, 1, 1)

st.set_page_config(page_title="PotholeIQ", layout="wide", page_icon="🕳️")
st.title("🕳️ PotholeIQ — Road Damage Analysis Pipeline")
st.caption("Upload a road image to run detection · segmentation · explanation · anomaly check")


# ─────────────────────────── Model loading (cached) ───────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models_loaded = {}

    # Classifier (try transfer first, fall back to CNN, then baseline)
    for name, ckpt_file, builder_fn in [
        ("transfer",   "transfer_best.pth",  _build_transfer),
        ("cnn",        "cnn_best.pth",        _build_cnn),
        ("baseline",   "baseline_best.pth",   _build_baseline),
    ]:
        ckpt = CKPT_DIR / ckpt_file
        if ckpt.exists():
            try:
                model = builder_fn()
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                model.eval().to(DEVICE)
                models_loaded["classifier"] = (model, name)
                break
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")

    # U-Net
    unet_ckpt = CKPT_DIR / "unet_best.pth"
    if unet_ckpt.exists():
        try:
            from src.models.unet import build_unet
            unet = build_unet()
            unet.load_state_dict(torch.load(unet_ckpt, map_location=DEVICE))
            unet.eval().to(DEVICE)
            models_loaded["unet"] = unet
        except Exception as e:
            st.warning(f"Could not load U-Net: {e}")

    # Autoencoder
    ae_ckpt = CKPT_DIR / "autoencoder_best.pth"
    if ae_ckpt.exists():
        try:
            from src.models.autoencoder import build_autoencoder
            ae = build_autoencoder()
            ae.load_state_dict(torch.load(ae_ckpt, map_location=DEVICE))
            ae.eval().to(DEVICE)
            models_loaded["autoencoder"] = ae
        except Exception as e:
            st.warning(f"Could not load Autoencoder: {e}")

    return models_loaded


def _build_transfer():
    from src.models.transfer import build_transfer_model
    return build_transfer_model(pretrained=False)

def _build_cnn():
    from src.models.cnn import build_cnn
    return build_cnn()

def _build_baseline():
    from src.models.baseline import build_baseline
    return build_baseline()


# ─────────────────────────── Image preprocessing ──────────────────────────────

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL image → normalised tensor (3, H, W)."""
    tf  = get_transforms("val")
    return tf(pil_img)


def denorm(tensor: torch.Tensor) -> np.ndarray:
    """Normalised tensor → uint8 numpy (H, W, 3)."""
    img = (tensor.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ─────────────────────────── Grad-CAM (inline) ───────────────────────────────

def run_gradcam(model, img_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Minimal Grad-CAM for the classifier.
    Returns heatmap np.ndarray (H, W) in [0, 1].
    """
    from src.explainability.gradcam import GradCAM

    # Identify the last conv layer
    target_layer = None
    if hasattr(model, "conv_blocks"):          # CustomCNN
        target_layer = model.conv_blocks[-1]
    elif hasattr(model, "backbone"):           # TransferModel
        target_layer = model.backbone.features[-1]
    if target_layer is None:
        return None

    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(img_tensor.unsqueeze(0).to(DEVICE), target_class)
    cam.remove_hooks()
    return heatmap


# ─────────────────────────── Severity label ───────────────────────────────────

def severity_badge(ratio: float) -> str:
    low  = CFG.unet.severity_thresholds.low_max
    med  = CFG.unet.severity_thresholds.medium_max
    if ratio < low:
        return "🟢 Low"
    elif ratio < med:
        return "🟡 Medium"
    else:
        return "🔴 High"


# ─────────────────────────── Main UI ─────────────────────────────────────────

uploaded = st.file_uploader("Choose a road image…", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("👆 Upload a road image to run the full pipeline.")
    st.stop()

pil_img    = Image.open(uploaded).convert("RGB")
img_tensor = pil_to_tensor(pil_img)
models     = load_models()

col_img, col_info = st.columns([1, 2])
with col_img:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Classifier
# ═══════════════════════════════════════════════════════════════════════════════
tab_cls, tab_seg, tab_xai, tab_ae = st.tabs(
    ["🔍 Classifier", "🗺️ Segmentation", "🔥 Grad-CAM", "🤖 Anomaly"]
)

with tab_cls:
    if "classifier" not in models:
        st.warning("⏳ No classifier checkpoint found. Run the training scripts first.")
    else:
        clf, clf_name = models["classifier"]
        with torch.no_grad():
            logits = clf(img_tensor.unsqueeze(0).to(DEVICE))
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        label     = int(probs.argmax())
        conf      = float(probs[label])
        cls_names = ["Normal Road", "Pothole Detected"]
        cls_color = "red" if label == 1 else "green"

        st.markdown(f"### Prediction: :{cls_color}[**{cls_names[label]}**]")
        st.markdown(f"**Confidence:** {conf:.1%}  |  **Model:** `{clf_name}`")

        st.progress(float(probs[1]), text=f"Pothole probability: {probs[1]:.1%}")

        col1, col2 = st.columns(2)
        col1.metric("Normal", f"{probs[0]:.1%}")
        col2.metric("Pothole", f"{probs[1]:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_seg:
    if "unet" not in models:
        st.warning("⏳ No U-Net checkpoint found. Run `python src/models/train_unet.py` first.")
    else:
        unet = models["unet"]
        with torch.no_grad():
            pred_mask = unet(img_tensor.unsqueeze(0).to(DEVICE))  # (1, 1, H, W)
            pred_mask = pred_mask.squeeze().cpu().numpy()           # (H, W)

        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        area_ratio  = binary_mask.mean()

        # Create overlay
        orig_np    = np.array(pil_img.resize((W, H)))
        mask_color = np.zeros_like(orig_np)
        mask_color[:, :, 0] = binary_mask * 255   # red channel
        overlay    = cv2.addWeighted(orig_np, 0.6, mask_color, 0.4, 0)

        col_a, col_b = st.columns(2)
        col_a.image(overlay, caption="Segmentation Overlay", use_container_width=True)
        col_b.image((pred_mask * 255).astype(np.uint8), caption="Raw Mask", use_container_width=True)

        sev = severity_badge(area_ratio)
        st.markdown(f"### Severity: {sev}")
        st.markdown(f"**Pothole area ratio:** {area_ratio:.2%}  "
                    f"(Low < 5% · Medium 5–20% · High > 20%)")
        st.caption("⚠️ Masks are generated from bounding-box annotations (rectangular approximation).")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_xai:
    if "classifier" not in models:
        st.warning("⏳ No classifier checkpoint found.")
    else:
        clf, clf_name = models["classifier"]
        # Only Grad-CAM-able models (not MLP baseline)
        if clf_name == "baseline":
            st.info("Grad-CAM requires a CNN or Transfer model. The baseline is an MLP — no spatial maps.")
        else:
            with torch.no_grad():
                logits = clf(img_tensor.unsqueeze(0).to(DEVICE))
                target = int(logits.argmax(1).item())

            try:
                heatmap = run_gradcam(clf, img_tensor, target)
                if heatmap is not None:
                    orig_np = np.array(pil_img.resize((W, H)))
                    # Apply jet colormap and blend
                    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    overlay       = cv2.addWeighted(orig_np, 0.5, heatmap_rgb, 0.5, 0)

                    col_a, col_b = st.columns(2)
                    col_a.image(orig_np,  caption="Original", use_container_width=True)
                    col_b.image(overlay,  caption="Grad-CAM Overlay", use_container_width=True)
                    st.caption(
                        "🔴 Hot regions = areas the model focuses on most for its prediction. "
                        "For potholes, activations should concentrate on the damaged road surface."
                    )
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Anomaly (Autoencoder)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ae:
    if "autoencoder" not in models:
        st.warning("⏳ No Autoencoder checkpoint found. Run `python src/models/train_autoencoder.py` first.")
    else:
        ae = models["autoencoder"]
        with torch.no_grad():
            x     = img_tensor.unsqueeze(0).to(DEVICE)
            recon = ae(x).squeeze(0).cpu()
            score = float(torch.mean((img_tensor - recon.cpu()) ** 2).item())

        threshold = CFG.autoencoder.anomaly_threshold
        is_anomaly = (threshold is not None) and (score > threshold)

        recon_np = denorm(recon)
        orig_np  = denorm(img_tensor)

        col_a, col_b = st.columns(2)
        col_a.image(orig_np,  caption="Original",      use_container_width=True)
        col_b.image(recon_np, caption="Reconstructed", use_container_width=True)

        if threshold is not None:
            status = "🔴 **Anomaly (Pothole)**" if is_anomaly else "🟢 **Normal Road**"
            st.markdown(f"### Anomaly verdict: {status}")
            st.markdown(f"**Reconstruction MSE:** `{score:.6f}`  |  "
                        f"**Threshold:** `{threshold:.6f}`")
        else:
            st.markdown(f"**Reconstruction MSE:** `{score:.6f}`")
            st.caption("Run train_autoencoder.py to compute and store the threshold.")

        st.caption(
            "The autoencoder was trained ONLY on normal road images. "
            "High reconstruction error indicates the input is unlike normal road textures (anomaly)."
        )

st.divider()
st.caption("PotholeIQ — B.Tech Deep Learning Course Project · Results are course-level")
