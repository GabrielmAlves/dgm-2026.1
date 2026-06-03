"""
HSV-based recoloration of segmented objects.

Pipeline B's image manipulation step. Takes an RGB image and a binary mask
identifying the object, replaces the hue inside the mask with a target color,
and preserves saturation and value (so shadows, highlights, and material
texture are kept). Outside the mask is untouched.

Design choices:

  * Substitute H, preserve S and V (not absolute color fill). The 'blue
    banana' must still look like a photograph — shadows on its sides, a
    specular highlight on its top — not a flat blue silhouette. Replacing
    the hue while keeping saturation and value approximates this.

  * Canonical color-to-hue map fixed in CANONICAL_HUE. Each of the 10
    colors used in the project maps to a single representative hue (in
    OpenCV's 0-179 range for uint8 HSV).

  * Special cases:
      - white: saturation forced to 0 (grayscale), value pushed to top
      - black: value forced near 0 (no hue makes sense for black)
      - brown: low saturation, mid-low value (brown is desaturated orange)
    These cannot be reached by hue substitution alone.

  * The segmentation step (Grounding DINO + SAM2) is in a separate class
    with lazy imports — it requires CUDA, transformers, and SAM2 packages
    that we don't want forced on every consumer of this module.

Tests in tests/test_segment_recolor.py validate the recoloration math
against hand-checked cases (red→blue shifts hue 180 degrees, white→...
goes to saturated colors, etc.) and the mask boundary handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from PIL.Image import Image

CANONICAL_HUE: dict[str, int] = {
    "red":    0,
    "orange": 13,    
    "yellow": 28,   
    "green":  60,    
    "blue":   110,   
    "purple": 140,   
    "pink":   163,   
}

ACHROMATIC = {"white", "black"}
DESATURATED = {"brown"}  

@dataclass(frozen=True)
class RecolorResult:
    """Outcome of a recoloration attempt."""
    image: np.ndarray              
    mask: np.ndarray               
    mask_area_frac: float          
    accepted: bool                 
    reason: str                    

def _rgb_to_hsv_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    Convert HxWx3 RGB uint8 to HxWx3 HSV uint8 (OpenCV convention: H in
    [0,179], S,V in [0,255]). Pure numpy, no opencv dependency.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask_delta = delta > 1e-8
    rc = cmax == r
    gc = cmax == g
    bc = cmax == b
    
    safe_delta = np.where(mask_delta, delta, 1.0)
    h_r = (60 * ((g - b) / safe_delta) + 360) % 360
    h_g = (60 * ((b - r) / safe_delta) + 120)
    h_b = (60 * ((r - g) / safe_delta) + 240)
    h = np.where(mask_delta & rc, h_r, h)
    h = np.where(mask_delta & ~rc & gc, h_g, h)
    h = np.where(mask_delta & ~rc & ~gc & bc, h_b, h)
    h = (h / 2.0).clip(0, 179)  

    s = np.where(cmax > 1e-8, delta / np.where(cmax > 1e-8, cmax, 1.0), 0.0) * 255
    v = cmax * 255

    return np.stack([h, s, v], axis=-1).clip(0, 255).astype(np.uint8)

def _hsv_to_rgb_uint8(hsv: np.ndarray) -> np.ndarray:
    """Inverse of _rgb_to_hsv_uint8. Pure numpy."""
    h = hsv[..., 0].astype(np.float32) * 2.0  
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    c = v * s
    x = c * (1 - np.abs(((h / 60.0) % 2) - 1))
    m = v - c

    rp = np.zeros_like(h); gp = np.zeros_like(h); bp = np.zeros_like(h)
    cond = (h < 60)
    rp = np.where(cond, c, rp); gp = np.where(cond, x, gp); bp = np.where(cond, 0, bp)
    cond = (h >= 60) & (h < 120)
    rp = np.where(cond, x, rp); gp = np.where(cond, c, gp); bp = np.where(cond, 0, bp)
    cond = (h >= 120) & (h < 180)
    rp = np.where(cond, 0, rp); gp = np.where(cond, c, gp); bp = np.where(cond, x, bp)
    cond = (h >= 180) & (h < 240)
    rp = np.where(cond, 0, rp); gp = np.where(cond, x, gp); bp = np.where(cond, c, bp)
    cond = (h >= 240) & (h < 300)
    rp = np.where(cond, x, rp); gp = np.where(cond, 0, gp); bp = np.where(cond, c, bp)
    cond = (h >= 300)
    rp = np.where(cond, c, rp); gp = np.where(cond, 0, gp); bp = np.where(cond, x, bp)

    r = ((rp + m) * 255).clip(0, 255)
    g = ((gp + m) * 255).clip(0, 255)
    b = ((bp + m) * 255).clip(0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def recolor_hsv(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_color: str,
    min_area: float = 0.05,
    max_area: float = 0.95,
    sat_boost: float = 1.2,
) -> RecolorResult:
    """
    Apply target_color to the masked region by substituting hue (and
    handling achromatic/desaturated special cases) in HSV space.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 array
    mask      : (H, W) bool or uint8 array; True/255 = recolor here
    target_color : one of CANONICAL_HUE keys ∪ ACHROMATIC ∪ DESATURATED
    min_area, max_area : reject if mask area fraction is outside [min, max]
    sat_boost : multiply saturation by this factor for chromatic colors,
                so that low-saturation objects (a beige chair) actually
                end up the requested color instead of a faint tint.
                Capped at 255.

    Returns
    -------
    RecolorResult — image always returned (caller decides what to do on
    rejection); accepted=False means area thresholds failed and the
    pipeline should discard this sample.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image_rgb.shape}")
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    if mask_bool.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} does not match image {image_rgb.shape[:2]}"
        )

    area_frac = float(mask_bool.mean())
    if area_frac < min_area:
        return RecolorResult(image=image_rgb.copy(), mask=mask_bool,
                             mask_area_frac=area_frac, accepted=False,
                             reason=f"mask too small ({area_frac:.1%} < {min_area:.1%})")
    if area_frac > max_area:
        return RecolorResult(image=image_rgb.copy(), mask=mask_bool,
                             mask_area_frac=area_frac, accepted=False,
                             reason=f"mask too large ({area_frac:.1%} > {max_area:.1%})")

    if target_color not in CANONICAL_HUE and target_color not in ACHROMATIC and target_color not in DESATURATED:
        raise ValueError(f"unknown target_color: {target_color}")

    hsv = _rgb_to_hsv_uint8(image_rgb)
    h, s, v = hsv[..., 0].copy(), hsv[..., 1].copy(), hsv[..., 2].copy()

    if target_color in CANONICAL_HUE:
        new_hue = CANONICAL_HUE[target_color]
        h[mask_bool] = new_hue
        s_boosted = np.clip(s.astype(np.float32) * sat_boost, 0, 255).astype(np.uint8)
        s[mask_bool] = np.maximum(s_boosted[mask_bool], 100)  # floor at 100/255
    elif target_color == "white":
        s[mask_bool] = 0
        v[mask_bool] = np.clip(v[mask_bool].astype(np.float32) * 1.3, 200, 255).astype(np.uint8)
    elif target_color == "black":
        s[mask_bool] = 0
        v[mask_bool] = np.clip(v[mask_bool].astype(np.float32) * 0.3, 0, 80).astype(np.uint8)
    elif target_color == "brown":
        h[mask_bool] = 13  
        s[mask_bool] = np.clip(s[mask_bool].astype(np.float32) * 0.6, 60, 180).astype(np.uint8)
        v[mask_bool] = np.clip(v[mask_bool].astype(np.float32) * 0.7, 60, 180).astype(np.uint8)

    out_hsv = np.stack([h, s, v], axis=-1)
    out_rgb = _hsv_to_rgb_uint8(out_hsv)
    return RecolorResult(image=out_rgb, mask=mask_bool,
                         mask_area_frac=area_frac, accepted=True, reason="ok")

class SegmentationPipeline:
    """
    Lazy wrapper around Grounding DINO + SAM2.

    Loads the models on first call (.segment), keeps them cached.
    Heavy imports are deferred so this module can be imported on a CPU-only
    machine for unit testing the recoloration math.
    """

    def __init__(
        self,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam2-hiera-large",
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.dino_model_id = dino_model_id
        self.sam_model_id = sam_model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._dino_processor = None
        self._dino_model = None
        self._sam_predictor = None

    def _ensure_loaded(self) -> None:
        if self._dino_model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_id
        ).to(self.device).eval()
        self._sam_predictor = SAM2ImagePredictor.from_pretrained(self.sam_model_id)
        self._torch = torch

    def segment(
        self,
        image: "Image",
        object_name: str | None = None,
        text_prompt: str | None = None,
    ) -> np.ndarray | None:
        """
        Detect and segment an object in `image`.

        Accepts either `object_name` (current style) or `text_prompt`
        (legacy keyword used by older callers). When both are passed,
        `object_name` wins. At least one must be provided.

        Returns a (H, W) bool mask of the highest-confidence detection, or
        None if no detection passed the thresholds. The caller can then pass
        this mask to recolor_hsv.

        Strategy: Grounding DINO finds the bounding box from the text prompt,
        SAM2 produces the pixel-accurate mask from the box.
        """
        if object_name is None and text_prompt is None:
            raise ValueError("segment() needs object_name or text_prompt")
        if object_name is None:
            object_name = text_prompt
        self._ensure_loaded()
        torch = self._torch

        text_prompt = f"a {object_name}."

        rgb_array = np.array(image.convert("RGB"))

        inputs = self._dino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._dino_model(**inputs)

        post_process = self._dino_processor.post_process_grounded_object_detection
        try:
            results = post_process(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except TypeError:
            results = post_process(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )
        if not results or len(results[0]["boxes"]) == 0:
            return None

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        best_idx = int(scores.argmax().item())
        best_box = boxes[best_idx].cpu().numpy()

        self._sam_predictor.set_image(rgb_array)
        masks, _, _ = self._sam_predictor.predict(
            box=best_box,
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None
        mask = masks[0].astype(bool)
        return mask
