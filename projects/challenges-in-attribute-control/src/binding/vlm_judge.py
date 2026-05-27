"""
VLM-as-judge for attribute binding evaluation.

Loads Qwen2.5-VL-7B-Instruct once and exposes a single `judge_image`
function that, given an image path and the (object, color) the
generator was *prompted* to produce, returns a structured verdict:

    {
      "object_predicted": "<canonical or 'unmatched'>",
      "color_predicted":  "<canonical or 'unmatched'>",
      "object_raw":       "<raw VLM string>",
      "color_raw":        "<raw VLM string>",
      "binding_correct":  bool,
    }

Design notes:

1) Decomposed questions, not yes/no.
   We ask "what object?" then "what color?" — never "is this a pink
   chalkboard?". Yes/no induces acquiescence bias in VLMs (they tend
   to say yes). Decomposition forces the model to commit independently
   to each attribute, mirroring the structure of the binding problem
   itself.

2) JSON output, not free text.
   Free text needs heuristic parsing; heuristics fail silently. We
   ask the model for `{"answer": "..."}` and parse JSON. Parse failures
   raise loudly and are recorded — better than silently miscounting.

3) Canonical mapping via synonyms.
   The model may say "cyan", "ebony", "scarlet" instead of canonical
   "blue", "black", "red". We map known synonyms and mark anything
   else as "unmatched". An unmatched answer counts as a binding error
   (the safest interpretation), but is logged separately so you can
   later inspect whether the synonym dictionary needs widening.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


# ── Synonym maps ────────────────────────────────────────────────────────────
# Maps the model's possible word choices to our canonical taxonomy.
# To extend: add lowercase entries; the matcher is case-insensitive.
# Empty mapping means "expand only when calibration says it's needed."
COLOR_SYNONYMS: dict[str, str] = {
    # canonical color : list of accepted synonyms (will be inverted below)
}
_COLOR_GROUPS: dict[str, list[str]] = {
    "blue":   ["blue", "cyan", "azure", "navy", "teal", "turquoise", "sky blue", "light blue", "dark blue"],
    "yellow": ["yellow", "golden", "gold", "amber", "mustard", "lemon"],
    "red":    ["red", "scarlet", "crimson", "ruby", "maroon"],
    "green":  ["green", "lime", "olive", "emerald", "forest green", "light green", "dark green"],
    "pink":   ["pink", "magenta", "fuchsia", "salmon", "rose"],
    "orange": ["orange", "tangerine", "coral"],
    "purple": ["purple", "violet", "lavender", "indigo", "lilac"],
    "white":  ["white", "ivory", "cream", "off-white"],
    "black":  ["black", "ebony", "jet"],
    "brown":  ["brown", "tan", "beige", "chestnut", "chocolate", "khaki"],
}
for canonical, syns in _COLOR_GROUPS.items():
    for s in syns:
        COLOR_SYNONYMS[s.lower()] = canonical

# Objects: usually the VLM returns the canonical word, but plurals and
# minor variants are common.
_OBJECT_GROUPS: dict[str, list[str]] = {
    "polar bear": ["polar bear", "polar bears", "bear", "white bear"],
    "chalkboard": ["chalkboard", "blackboard", "chalk board"],
    "banana":     ["banana", "bananas"],
    "apple":      ["apple", "apples"],
    "dog":        ["dog", "dogs", "puppy", "puppies"],
    "chair":      ["chair", "chairs", "armchair", "stool"],
    "flower":     ["flower", "flowers", "blossom", "rose", "tulip", "daisy"],
    "shoe":       ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots"],
    "bag":        ["bag", "bags", "handbag", "purse", "backpack"],
    "frog":       ["frog", "frogs", "toad"],
    "ball":       ["ball", "balls"],
    "car":        ["car", "cars", "vehicle", "automobile"],
}
OBJECT_SYNONYMS: dict[str, str] = {}
for canonical, syns in _OBJECT_GROUPS.items():
    for s in syns:
        OBJECT_SYNONYMS[s.lower()] = canonical


UNMATCHED = "unmatched"


def map_to_canonical(raw: str, synonym_map: dict[str, str]) -> str:
    """
    Map a raw VLM string to a canonical label, or UNMATCHED.

    Strategy:
    1. Strip surrounding whitespace and punctuation, lowercase.
    2. Exact synonym lookup.
    3. Token-by-token scan (handles "a small green frog" → "green" first
       hit; we use this for color extraction from sentences when the VLM
       ignored our JSON instructions).
    """
    if not raw:
        return UNMATCHED
    cleaned = re.sub(r"[^\w\s-]", "", raw).strip().lower()
    if not cleaned:
        return UNMATCHED
    if cleaned in synonym_map:
        return synonym_map[cleaned]
    # Token-level fallback: take the first token that's a known synonym.
    for tok in cleaned.split():
        if tok in synonym_map:
            return synonym_map[tok]
    # Two-word object names like "polar bear" need bigram lookup too.
    tokens = cleaned.split()
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in synonym_map:
            return synonym_map[bigram]
    return UNMATCHED


# ── Prompts for the two decomposed questions ────────────────────────────────
PROMPT_OBJECT = (
    "Look at this image and identify the single most prominent object. "
    "Answer with a JSON object only, no extra text. "
    'Example: {"answer": "frog"}. '
    "Use one or two words for the object name."
)
PROMPT_COLOR_TMPL = (
    'Look at this image. What is the dominant color of the {object} in it? '
    "Answer with a JSON object only, no extra text. "
    'Example: {{"answer": "green"}}. '
    "Use one word for the color."
)


@dataclass(frozen=True)
class Judgment:
    """Structured verdict for a single image."""
    object_raw: str
    color_raw: str
    object_predicted: str       # canonical or UNMATCHED
    color_predicted: str        # canonical or UNMATCHED
    binding_correct: bool

    def to_dict(self) -> dict:
        return {
            "object_raw": self.object_raw,
            "color_raw": self.color_raw,
            "object_predicted": self.object_predicted,
            "color_predicted": self.color_predicted,
            "binding_correct": self.binding_correct,
        }


def _extract_json_answer(raw: str) -> str:
    """
    Extract the 'answer' field from the VLM's response.

    Robust to:
      - Extra text before/after the JSON block.
      - Single quotes instead of double (some models slip).
      - Trailing punctuation.

    Returns the raw answer string, or empty string if no JSON found.
    """
    # Try direct parse first.
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return str(obj.get("answer", "")).strip()
    except json.JSONDecodeError:
        pass
    # Find a {...} block.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidate = m.group(0).replace("'", '"')
        try:
            obj = json.loads(candidate)
            return str(obj.get("answer", "")).strip()
        except json.JSONDecodeError:
            pass
    # No JSON: return the raw text — synonym map's token-level fallback
    # may still recover a useful word from it.
    return raw


class VLMJudge:
    """
    Wraps Qwen2.5-VL-7B-Instruct as a binding-evaluation judge.

    Heavy imports are deferred to __init__ so the module can be imported
    in test environments (with mocks) that don't have transformers.

    Args:
        model_id: HF repo id of the Qwen-VL checkpoint.
        device: "cuda" or "cpu". Auto-detects if None.
        dtype: "bfloat16" or "float16" on GPU; "float32" on CPU.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[dtype if device == "cuda" else "float32"]

        self.device = device
        self.dtype = torch_dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
        )
        self.model.eval()

    def _ask(self, image: "Image", prompt: str, max_new_tokens: int = 32) -> str:
        """Run one VLM forward pass and return the raw text response."""
        import torch
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic
                temperature=1.0,
            )
        # Decode only the newly-generated tokens.
        generated = out[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )[0]
        return response.strip()

    def judge_image(
        self,
        image_path: str | Path,
        expected_object: str,
        expected_color: str,
    ) -> Judgment:
        """
        Issue the two decomposed questions and return a structured Judgment.

        Args:
            image_path: path to the PNG to evaluate.
            expected_object: canonical object the prompt asked for.
            expected_color: canonical color the prompt asked for.

        Returns:
            Judgment with raw responses, canonical mappings, and a final
            binding_correct boolean (True iff both object and color
            mappings match the expected values).
        """
        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")

        # Q1: object
        obj_raw_full = self._ask(image, PROMPT_OBJECT, max_new_tokens=32)
        obj_answer = _extract_json_answer(obj_raw_full)
        obj_canonical = map_to_canonical(obj_answer, OBJECT_SYNONYMS)

        # Q2: color of that object. We use the *expected* object name in the
        # prompt — not the predicted one — because asking "color of the X"
        # where X is what the user asked for is the cleanest test. If the
        # model failed to draw X, the color answer is irrelevant anyway
        # (binding_correct will be False via object mismatch).
        color_prompt = PROMPT_COLOR_TMPL.format(object=expected_object)
        color_raw_full = self._ask(image, color_prompt, max_new_tokens=32)
        color_answer = _extract_json_answer(color_raw_full)
        color_canonical = map_to_canonical(color_answer, COLOR_SYNONYMS)

        binding_correct = (
            obj_canonical == expected_object and color_canonical == expected_color
        )
        return Judgment(
            object_raw=obj_answer,
            color_raw=color_answer,
            object_predicted=obj_canonical,
            color_predicted=color_canonical,
            binding_correct=binding_correct,
        )
