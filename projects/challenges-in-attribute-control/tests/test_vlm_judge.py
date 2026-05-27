"""
Tests for binding.vlm_judge.

The heavy VLM (Qwen2.5-VL-7B, ~15 GB) is mocked. We test the *logic*
that surrounds it
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from binding.vlm_judge import (
    COLOR_SYNONYMS,
    OBJECT_SYNONYMS,
    UNMATCHED,
    Judgment,
    VLMJudge,
    _extract_json_answer,
    map_to_canonical,
)


# ── Synonym mapping ─────────────────────────────────────────────────────────

def test_canonical_color_passthrough():
    assert map_to_canonical("green", COLOR_SYNONYMS) == "green"
    assert map_to_canonical("BLUE", COLOR_SYNONYMS) == "blue"


def test_color_synonyms_resolve():
    assert map_to_canonical("cyan", COLOR_SYNONYMS) == "blue"
    assert map_to_canonical("scarlet", COLOR_SYNONYMS) == "red"
    assert map_to_canonical("ebony", COLOR_SYNONYMS) == "black"
    assert map_to_canonical("tan", COLOR_SYNONYMS) == "brown"


def test_color_unmatched():
    assert map_to_canonical("octarine", COLOR_SYNONYMS) == UNMATCHED
    assert map_to_canonical("", COLOR_SYNONYMS) == UNMATCHED
    assert map_to_canonical("???", COLOR_SYNONYMS) == UNMATCHED


def test_color_in_sentence_extracted():
    assert map_to_canonical("The frog is green", COLOR_SYNONYMS) == "green"


def test_object_bigram_polar_bear():
    """'polar bear' is two words; bigram lookup must work."""
    assert map_to_canonical("polar bear", OBJECT_SYNONYMS) == "polar bear"
    assert map_to_canonical("a polar bear", OBJECT_SYNONYMS) == "polar bear"
    assert map_to_canonical("bear", OBJECT_SYNONYMS) == "polar bear"


def test_object_plurals():
    assert map_to_canonical("bananas", OBJECT_SYNONYMS) == "banana"
    assert map_to_canonical("dogs", OBJECT_SYNONYMS) == "dog"


# ── JSON extraction ─────────────────────────────────────────────────────────

def test_extract_json_clean():
    assert _extract_json_answer('{"answer": "green"}') == "green"


def test_extract_json_with_surrounding_text():
    raw = 'Sure! Here is my answer: {"answer": "blue"}. Hope it helps!'
    assert _extract_json_answer(raw) == "blue"


def test_extract_json_single_quotes():
    """Some models slip and use single quotes — should still parse."""
    assert _extract_json_answer("{'answer': 'red'}") == "red"


def test_extract_no_json_returns_raw():
    """Fallback: raw text goes back so synonym matcher can try."""
    assert _extract_json_answer("The frog is green") == "The frog is green"


# ── Judgment dataclass ─────────────────────────────────────────────────────

def test_judgment_to_dict():
    j = Judgment(
        object_raw="frog", color_raw="green",
        object_predicted="frog", color_predicted="green",
        binding_correct=True,
    )
    d = j.to_dict()
    assert d["binding_correct"] is True
    assert d["object_predicted"] == "frog"
    assert d["color_predicted"] == "green"


def test_judgment_frozen():
    """Judgment is immutable — protects against accidental mutation downstream."""
    j = Judgment("a", "b", "a", "b", True)
    try:
        j.binding_correct = False 
        raised = False
    except Exception:
        raised = True
    assert raised


# ── End-to-end judge_image with mocked VLM ─────────────────────────────────

def _make_mock_judge(obj_response: str, color_response: str) -> VLMJudge:
    """Construct a VLMJudge with __init__ skipped and _ask mocked."""
    with patch.object(VLMJudge, "__init__", return_value=None):
        judge = VLMJudge()  
    judge.device = "cpu"   
    judge._ask = MagicMock(side_effect=[obj_response, color_response])  
    return judge


def test_judge_image_correct_binding(tmp_path):
    """Both questions answered correctly → binding_correct=True."""
    from PIL import Image as PILImage
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (4, 4), (0, 0, 0)).save(img_path)

    judge = _make_mock_judge('{"answer": "frog"}', '{"answer": "green"}')
    j = judge.judge_image(img_path, expected_object="frog", expected_color="green")
    assert j.object_predicted == "frog"
    assert j.color_predicted == "green"
    assert j.binding_correct is True


def test_judge_image_wrong_color(tmp_path):
    """Object right, color wrong → binding_correct=False."""
    from PIL import Image as PILImage
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (4, 4), (0, 0, 0)).save(img_path)

    judge = _make_mock_judge('{"answer": "frog"}', '{"answer": "red"}')
    j = judge.judge_image(img_path, expected_object="frog", expected_color="green")
    assert j.object_predicted == "frog"
    assert j.color_predicted == "red"
    assert j.binding_correct is False


def test_judge_image_synonym_accepted(tmp_path):
    """VLM saying 'cyan' for an expected 'blue' should still count as correct."""
    from PIL import Image as PILImage
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (4, 4), (0, 0, 0)).save(img_path)

    judge = _make_mock_judge('{"answer": "car"}', '{"answer": "cyan"}')
    j = judge.judge_image(img_path, expected_object="car", expected_color="blue")
    assert j.color_predicted == "blue"
    assert j.binding_correct is True


def test_judge_image_unmatched_is_incorrect(tmp_path):
    """Unrecognized color → unmatched → binding_correct=False."""
    from PIL import Image as PILImage
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (4, 4), (0, 0, 0)).save(img_path)

    judge = _make_mock_judge('{"answer": "frog"}', '{"answer": "octarine"}')
    j = judge.judge_image(img_path, expected_object="frog", expected_color="green")
    assert j.color_predicted == UNMATCHED
    assert j.binding_correct is False
