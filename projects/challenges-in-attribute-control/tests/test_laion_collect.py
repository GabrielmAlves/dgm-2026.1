"""
Tests for binding.laion_collect.

Network (download_image) is not tested here — it depends on live URLs.
We test the candidate-selection logic, which is where a bug would
silently poison the training set with wrong images.
"""

from __future__ import annotations

from binding.laion_collect import (
    Candidate,
    CollectionTargets,
    caption_mentions,
    is_gap_candidate,
    iter_candidates,
)

def fake_binding(caption: str, obj: str, color: str) -> bool:
    return f"{color} {obj}" in caption.lower()

def test_caption_mentions_case_insensitive():
    assert caption_mentions("A Blue Banana", "blue")
    assert caption_mentions("A Blue Banana", "banana")
    assert not caption_mentions("a red car", "banana")
    assert not caption_mentions("", "banana")


def test_gap_candidate_accepts_unbound_cooccurrence():
    """Both words present, no syntactic binding → gap candidate (accept)."""
    cap = "banana, fruit, color: orange, fresh"
    assert is_gap_candidate(cap, "banana", "orange", fake_binding)


def test_gap_candidate_rejects_bound_caption():
    """If 'orange banana' already appears, caption is good — NOT a gap case."""
    cap = "a fresh orange banana on a table"
    assert not is_gap_candidate(cap, "banana", "orange", fake_binding)


def test_gap_candidate_rejects_missing_word():
    """Missing the color → not a candidate."""
    cap = "a banana on a table"
    assert not is_gap_candidate(cap, "banana", "orange", fake_binding)


def test_collection_targets_accounting():
    t = CollectionTargets(needed={("banana", "orange"): 2, ("apple", "blue"): 1})
    assert t.remaining() == 3
    assert t.want("banana", "orange")
    t.record("banana", "orange")
    assert t.needed[("banana", "orange")] == 1
    assert t.remaining() == 2
    t.record("banana", "orange")
    assert not t.want("banana", "orange")  
    assert t.remaining() == 1


def test_iter_candidates_respects_targets():
    """The generator should stop nominating a pair once its target is met."""
    rows = [
        {"caption": "banana, color orange, fresh", "url": "u1", "width": 500, "height": 500},
        {"caption": "orange banana ripe", "url": "u2"}, 
        {"caption": "banana photo, orange background", "url": "u3"},  
        {"caption": "banana, orange tag", "url": "u4"},  
    ]
    targets = CollectionTargets(needed={("banana", "orange"): 2})
    got = list(iter_candidates(rows, targets, fake_binding, max_scan=100))
    
    assert len(got) == 2
    assert all(isinstance(c, Candidate) for c in got)
    assert all(c.object_name == "banana" and c.color == "orange" for c in got)
    assert "u2" not in [c.url for c in got]


def test_iter_candidates_max_scan_cap():
    """If max_scan is hit before targets are met, generator stops anyway."""
    rows = [{"caption": "nothing relevant here", "url": f"u{i}"} for i in range(10)]
    targets = CollectionTargets(needed={("banana", "orange"): 5})
    got = list(iter_candidates(rows, targets, fake_binding, max_scan=10))
    assert len(got) == 0  


def test_iter_candidates_multiple_pairs():
    """Generator handles multiple needed pairs in one stream."""
    rows = [
        {"caption": "banana with orange label", "url": "u1"},
        {"caption": "apple, blue box behind", "url": "u2"},
    ]
    targets = CollectionTargets(needed={("banana", "orange"): 1, ("apple", "blue"): 1})
    got = list(iter_candidates(rows, targets, fake_binding, max_scan=100))
    pairs = {(c.object_name, c.color) for c in got}
    assert pairs == {("banana", "orange"), ("apple", "blue")}

from binding.laion_collect import is_bound_candidate  

def test_bound_candidate_accepts_bound_caption():
    """A caption with explicit binding ('orange banana') is a bound candidate."""
    cap = "a fresh orange banana on a table"
    assert is_bound_candidate(cap, "banana", "orange", fake_binding)


def test_bound_candidate_rejects_unbound_cooccurrence():
    """If words appear but without binding pattern, it's NOT a bound candidate."""
    cap = "banana, color orange, fresh"
    assert not is_bound_candidate(cap, "banana", "orange", fake_binding)


def test_bound_and_gap_are_disjoint():
    """For the same (caption, obj, color), at most one predicate fires."""
    captions = [
        "a fresh orange banana",         
        "banana, color orange, fresh",   
        "apple pie no banana",           
        "orange juice no fruit",         
    ]
    for cap in captions:
        g = is_gap_candidate(cap, "banana", "orange", fake_binding)
        b = is_bound_candidate(cap, "banana", "orange", fake_binding)
        assert not (g and b), f"both predicates fired for: {cap!r}"

def test_iter_candidates_bound_mode():
    """Using is_bound_candidate as predicate harvests the bound captions, not gaps."""
    from binding.laion_collect import is_bound_candidate
    rows = [
        {"caption": "orange banana ripe",          "url": "u1"},  
        {"caption": "banana, color orange",        "url": "u2"},  
        {"caption": "a fresh orange banana",       "url": "u3"},  
    ]
    targets = CollectionTargets(needed={("banana", "orange"): 5})
    got = list(iter_candidates(rows, targets, fake_binding,
                                max_scan=100, predicate=is_bound_candidate))
    urls = [c.url for c in got]
    assert "u2" not in urls
    assert "u1" in urls and "u3" in urls
