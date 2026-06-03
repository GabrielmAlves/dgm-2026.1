"""
LAION candidate collection for the finetuning dataset (Phase 5, Pipeline A).

Finds images in LAION-400M whose caption mentions both an object and a
color but WITHOUT syntactic binding — i.e. exactly the "gap" measured in
Experiment 1. These are likely-correct images with badly-formatted
captions: the raw material for re-captioning.

CRITICAL design rule: the original LAION caption is used ONLY as a coarse
search filter. It is never propagated to the training set. Ground truth
about what the image actually shows comes later, from the VLM (Pipeline A
step 2), never from this caption. This module just nominates candidates.

LAION-400M stores URLs, not images. Many URLs are dead (link rot affects
~30-50% of LAION-400M). The downloader is therefore tolerant: timeouts,
retries, and content-type / size checks, logging failures instead of
crashing.

Typical usage is via experiments/exp5_collect_laion.py, not directly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class Candidate:
    """A LAION row nominated as a possible training image for a pair."""
    object_name: str
    color: str
    url: str
    original_caption: str           
    width: int | None = None
    height: int | None = None

@dataclass
class CollectionTargets:
    """How many candidates we still need per (object, color) pair."""
    needed: dict[tuple[str, str], int] = field(default_factory=dict)

    def remaining(self) -> int:
        return sum(max(0, v) for v in self.needed.values())

    def want(self, obj: str, color: str) -> bool:
        return self.needed.get((obj, color), 0) > 0

    def record(self, obj: str, color: str) -> None:
        key = (obj, color)
        if key in self.needed:
            self.needed[key] -= 1

def caption_mentions(caption: str, word: str) -> bool:
    """Case-insensitive whole-word-ish membership (mirrors Exp1 contains_word)."""
    if not caption:
        return False
    return word.lower() in caption.lower()

def is_gap_candidate(
    caption: str,
    obj: str,
    color: str,
    binding_fn,
) -> bool:
    """
    A caption is a 'gap candidate' for (obj, color) when BOTH words appear
    but the syntactic binding pattern does NOT fire. That's the population
    Experiment 1 counted as co-occurring-without-binding — images likely
    showing the pair, captioned in a way that loses the binding signal.

    Used for Phase 5 treated/held_out collection: these are the cases the
    finetuning dataset will fix by re-captioning via VLM ground truth.

    binding_fn is the object_color_pair function from Experiment 1
    (injected to keep this module decoupled from the regex implementation).
    """
    cl = caption.lower()
    if not (caption_mentions(cl, obj) and caption_mentions(cl, color)):
        return False
    
    if binding_fn(cl, obj, color):
        return False
    return True


def is_bound_candidate(
    caption: str,
    obj: str,
    color: str,
    binding_fn,
) -> bool:
    """
    A caption is a 'bound candidate' for (obj, color) when both words appear
    AND the syntactic binding pattern fires. These are well-formatted captions
    like "a red apple", "the green frog" — exactly what's wanted for the
    control set: canonical pairs the model should keep getting right
    after finetuning (anchors against catastrophic forgetting).

    The image still needs VLM verification — captioning errors exist even in
    bound captions — but the prior on these is very high: the LAION caption
    structure matches the truth more often than not.
    """
    cl = caption.lower()
    if not (caption_mentions(cl, obj) and caption_mentions(cl, color)):
        return False
    return binding_fn(cl, obj, color)


def iter_candidates(
    dataset,
    targets: CollectionTargets,
    binding_fn,
    max_scan: int = 2_000_000,
    url_field: str = "url",
    caption_field: str = "caption",
    predicate=None,
) -> Iterator[Candidate]:
    """
    Stream the LAION dataset, yielding Candidates for pairs still needed.

    Stops when either `targets.remaining()` hits zero or `max_scan` rows
    have been inspected (a safety cap so a rare pair doesn't scan forever).

    `predicate` is the function that decides if a (caption, obj, color)
    triple is a candidate. Defaults to is_gap_candidate (Phase 5 default).
    Pass is_bound_candidate for the control-set collection.

    The caller is responsible for downloading and VLM-verifying each
    candidate; this generator only nominates them by caption.
    """
    if predicate is None:
        predicate = is_gap_candidate
    objects = sorted({o for (o, _c) in targets.needed})
    colors = sorted({c for (_o, c) in targets.needed})

    scanned = 0
    for row in dataset:
        if targets.remaining() <= 0 or scanned >= max_scan:
            break
        scanned += 1
        caption = row.get(caption_field)
        if not caption:
            continue
        cl = caption.lower()
        
        if not any(caption_mentions(cl, o) for o in objects):
            continue
        if not any(caption_mentions(cl, c) for c in colors):
            continue
        
        for obj in objects:
            if not caption_mentions(cl, obj):
                continue
            for color in colors:
                if not targets.want(obj, color):
                    continue
                if predicate(caption, obj, color, binding_fn):
                    targets.record(obj, color)
                    yield Candidate(
                        object_name=obj,
                        color=color,
                        url=row.get(url_field, ""),
                        original_caption=caption,
                        width=row.get("width"),
                        height=row.get("height"),
                    )
                    break  


def download_image(
    url: str,
    timeout: float = 10.0,
    max_bytes: int = 12_000_000,
    min_side: int = 128,
) -> "Image | None":
    """
    Download and decode an image URL. Returns a PIL image or None on any
    failure (dead link, timeout, non-image content, too small, too large).

    Tolerant by design: LAION link rot means many URLs 404. We never raise
    on a single bad URL — we just return None and let the caller move on.
    """
    import requests
    from PIL import Image as PILImage

    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            headers={"User-Agent": "Mozilla/5.0 (research dataset collection)"})
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            return None
        
        data = b""
        for chunk in resp.iter_content(8192):
            data += chunk
            if len(data) > max_bytes:
                return None
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        if min(img.size) < min_side:
            return None
        return img
    except Exception:
        return None
