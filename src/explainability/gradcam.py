"""
src/explainability/gradcam.py
==============================
Manual Grad-CAM implementation — built from scratch so every step can be
explained in a viva without relying on a third-party library.

Theory (Selvaraju et al., 2017):
  1. Run a forward pass and record the activations A^k (feature maps) at the
     chosen convolutional layer (the last one captures the richest semantics).
  2. Run a backward pass for the target class score and record ∂y^c / ∂A^k.
  3. Global-average-pool the gradients over spatial dimensions to get a scalar
     weight α_k^c for each feature-map channel k.
  4. Compute the weighted sum Σ_k α_k^c · A^k -> raw heatmap (H_raw).
  5. Apply ReLU: only pixels where the class has a *positive* influence are kept.
  6. Normalise to [0, 1] for display.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Convert a normalised CHW tensor back to a uint8 HWC numpy image.

    Args:
        tensor: (3, H, W) float32 tensor normalised with *mean* and *std*.
        mean:   sequence of 3 floats (e.g. [0.485, 0.456, 0.406])
        std:    sequence of 3 floats (e.g. [0.229, 0.224, 0.225])

    Returns:
        np.ndarray of shape (H, W, 3) dtype uint8, values in [0, 255].
    """
    mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(std,  dtype=np.float32).reshape(3, 1, 1)

    img = tensor.cpu().float().numpy()   # (3, H, W)
    img = img * std + mean               # undo normalisation
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))   # -> (H, W, 3) RGB
    return img


# ---------------------------------------------------------------------------
# GradCAM class
# ---------------------------------------------------------------------------
class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Usage:
        gcam = GradCAM(model, model.conv_blocks[-1])
        heatmap = gcam.generate(input_tensor, target_class=1)
        overlay = gcam.overlay_on_image(original_img_uint8, heatmap)
        gcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model:        The trained nn.Module (e.g. CustomCNN).
            target_layer: Layer whose activations and gradients to capture.
                          Typically the last convolutional block.
        """
        self.model = model
        self.target_layer = target_layer

        # Storage buffers filled by the hooks
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        # Register hooks — forward hook saves the output tensor of the layer,
        # backward hook saves the gradient flowing back through it.
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # --- Hook callbacks ----------------------------------------------------- #

    def _save_activation(self, module, input, output):
        """Forward hook: cache the layer's output (feature maps)."""
        # Detach so the saved tensor doesn't hold the full computation graph
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: cache the gradient w.r.t. the layer's output."""
        # grad_output[0] is ∂loss / ∂layer_output
        self._gradients = grad_output[0].detach()

    # --- Core algorithm ----------------------------------------------------- #

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Produce a Grad-CAM heatmap for *input_tensor*.

        Args:
            input_tensor: (1, C, H, W) pre-processed input (batch size 1).
            target_class: Class index to explain.  If None, uses the predicted
                          class (argmax of logits).

        Returns:
            np.ndarray of shape (H_layer, W_layer) in [0, 1].
        """
        self.model.eval()

        # 1. Forward pass — also fills self._activations via the hook
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(False)   # We'll zero out model grads only

        logits = self.model(input_tensor)    # (1, num_classes)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # 2. Backward pass for the target class score
        self.model.zero_grad()
        # We backpropagate through just the target-class score (not softmax
        # output) so the gradient reflects the raw class influence.
        score = logits[0, target_class]
        score.backward()
        # self._gradients now holds ∂score / ∂A  (B, C, H, W) with B=1

        # 3. Global-average-pool the gradients over spatial dims -> weights
        #    α_k = (1/Z) Σ_{i,j} (∂y^c / ∂A^k_{ij})
        grads = self._gradients.squeeze(0)   # (C, H, W)
        weights = grads.mean(dim=(1, 2))     # (C,)

        # 4. Weighted combination of activation maps
        activations = self._activations.squeeze(0)  # (C, H, W)
        # Multiply each channel map by its scalar weight, then sum
        heatmap = torch.einsum("c,chw->hw", weights, activations)  # (H, W)
        heatmap = heatmap.cpu().numpy()

        # 5. ReLU — discard negative contributions (they suppress the class)
        heatmap = np.maximum(heatmap, 0)

        # 6. Normalise to [0, 1] for overlay
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:                      # avoid divide-by-zero
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)   # degenerate case

        return heatmap  # (H_layer, W_layer) ∈ [0, 1]

    # --- Overlay helper ----------------------------------------------------- #

    def overlay_on_image(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Blend a Grad-CAM heatmap onto the original image.

        Args:
            original_img: (H, W, 3) uint8 RGB image.
            heatmap:      (h, w) float array in [0, 1].
            alpha:        Blending weight for the heatmap overlay.

        Returns:
            (H, W, 3) uint8 RGB composite.
        """
        H, W = original_img.shape[:2]

        # Resize heatmap to match input image resolution
        heatmap_resized = cv2.resize(heatmap, (W, H))  # (H, W) float

        # Map [0,1] -> [0,255] and apply jet colormap (OpenCV uses BGR)
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend: overlay = α * heatmap + (1 - α) * original
        overlay = (
            alpha * heatmap_color_rgb.astype(np.float32)
            + (1.0 - alpha) * original_img.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    # --- Cleanup ------------------------------------------------------------ #

    def remove_hooks(self):
        """Remove registered hooks to prevent memory leaks.

        Always call this after you are done generating heatmaps to allow
        the model to be garbage-collected normally.
        """
        self._fwd_hook.remove()
        self._bwd_hook.remove()
