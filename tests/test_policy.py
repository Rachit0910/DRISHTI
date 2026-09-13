"""Unit tests for the narration policy.

These run anywhere — no camera, no NPU, no speaker — because the policy is pure.
That is deliberate: the part of Drishti most likely to be wrong is the part that
decides what to say, so it is the part that must be testable on any machine.

    python -m pytest tests/ -q        (or)        python tests/test_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.phrases import ENGLISH, HINDI, get  # noqa: E402
from drishti.policy import (  # noqa: E402
    NarrationPolicy,
    Observation,
    bearing_of,
    distance_phrase,
    salience_of,
)


def obs(label: str, distance_m: float, cx: float = 0.5, conf: float = 0.9) -> Observation:
    half = 0.08
    return Observation(
        label=label,
        confidence=conf,
        bbox=(cx - half, 0.4, cx + half, 0.9),
        distance_m=distance_m,
    )


def test_bearing_buckets():
    assert bearing_of(0.10) == "to your left"
    assert bearing_of(0.35) == "slightly left"
    assert bearing_of(0.50) == "ahead"
    assert bearing_of(0.65) == "slightly right"
    assert bearing_of(0.95) == "to your right"


def test_distance_phrasing():
    assert distance_phrase(0.6) == "right in front of you"
    assert distance_phrase(1.2) == "one metre"
    assert distance_phrase(2.6) == "3 metres"
    assert distance_phrase(9.0) == "further ahead"


def test_closer_is_more_salient():
    assert salience_of(obs("person", 1.0)) > salience_of(obs("person", 4.0))


def test_central_is_more_salient_than_peripheral():
    assert salience_of(obs("person", 2.0, cx=0.5)) > salience_of(obs("person", 2.0, cx=0.95))


def test_ignored_classes_score_zero():
    assert salience_of(obs("laptop", 0.5)) == 0.0


def test_silent_when_nothing_matters():
    policy = NarrationPolicy()
    assert policy.tick([], now=0.0) == []
    # A distant, low-hazard object is below the salience floor.
    assert policy.tick([obs("potted plant", 12.0, cx=0.05)], now=1.0) == []


def test_speaks_once_then_suppresses_repeat():
    policy = NarrationPolicy()
    first = policy.tick([obs("person", 2.0)], now=0.0)
    assert len(first) == 1
    assert "Person" in first[0].text
    assert first[0].interrupt is False

    # Same person, next frames, same distance band: nothing new to say.
    assert policy.tick([obs("person", 2.1)], now=0.2) == []
    assert policy.tick([obs("person", 2.4)], now=0.5) == []
    # Drifting further away is not actionable either.
    assert policy.tick([obs("person", 4.0)], now=0.8) == []


def test_depth_jitter_across_a_band_boundary_does_not_retrigger():
    """A depth estimate wobbling around 2.0 m must not re-announce the same object."""
    policy = NarrationPolicy()
    assert len(policy.tick([obs("chair", 2.05)], now=0.0)) == 1
    for i, d in enumerate([1.98, 2.03, 1.97, 2.06, 1.99], start=1):
        assert policy.tick([obs("chair", d)], now=i * 0.1) == [], f"retriggered at {d} m"


def test_reannounces_when_object_gets_closer():
    policy = NarrationPolicy()
    assert len(policy.tick([obs("chair", 3.0)], now=0.0)) == 1
    # Crossing into a nearer distance band is new, actionable information.
    said = policy.tick([obs("chair", 1.5)], now=1.0)
    assert len(said) == 1


def test_critical_hazard_interrupts_and_leads_with_stop():
    policy = NarrationPolicy()
    said = policy.tick([obs("car", 0.9)], now=0.0)
    assert len(said) == 1
    assert said[0].interrupt is True
    assert said[0].text.startswith("Stop.")


def test_critical_hazard_bypasses_the_budget():
    policy = NarrationPolicy(max_utterances=1, window_s=60.0)
    assert len(policy.tick([obs("chair", 2.0, cx=0.5)], now=0.0)) == 1
    # Budget is now spent for the window.
    assert policy.tick([obs("bench", 2.0, cx=0.2)], now=1.0) == []
    # ...but a car about to hit you still gets through.
    said = policy.tick([obs("car", 0.8)], now=2.0)
    assert len(said) == 1 and said[0].interrupt


def test_budget_caps_chatter_then_recovers():
    policy = NarrationPolicy(max_utterances=2, window_s=10.0, repeat_cooldown_s=0.0)
    spoken = 0
    for i in range(6):
        spoken += len(policy.tick([obs("chair", 2.0, cx=0.3 + i * 0.02)], now=float(i)))
    assert spoken == 2, f"expected the budget to cap output at 2, got {spoken}"

    # After the window slides past, speech is allowed again.
    later = policy.tick([obs("chair", 2.0, cx=0.5)], now=30.0)
    assert len(later) == 1


def test_only_one_routine_utterance_per_frame():
    policy = NarrationPolicy()
    crowded = [obs("person", 1.8, cx=0.3), obs("chair", 2.0, cx=0.6), obs("bench", 2.2, cx=0.8)]
    said = policy.tick(crowded, now=0.0)
    assert len(said) == 1, "speech is serial; never queue three sentences from one frame"


def test_most_salient_object_wins_the_channel():
    policy = NarrationPolicy()
    said = policy.tick([obs("bench", 4.0, cx=0.9), obs("person", 1.3, cx=0.5)], now=0.0)
    assert len(said) == 1
    assert "Person" in said[0].text


def test_no_user_facing_string_is_hardcoded_in_the_policy():
    """Localisation must be a data change. If this fails, a string leaked into logic."""
    policy = NarrationPolicy(phrases=HINDI)
    said = policy.tick([obs("person", 2.0)], now=0.0)
    assert len(said) == 1
    text = said[0].text
    assert not any(word in text for word in ("metre", "ahead", "left", "right")), text
    assert "मीटर" in text


def test_critical_template_is_localised_too():
    policy = NarrationPolicy(phrases=HINDI)
    said = policy.tick([obs("car", 0.8)], now=0.0)
    assert len(said) == 1 and said[0].interrupt
    assert not said[0].text.startswith("Stop.")
    assert "रुकिए" in said[0].text


def test_unknown_locale_falls_back_to_english():
    assert get("fr") is ENGLISH
    assert get("hi-IN") is HINDI
    assert get("en-GB") is ENGLISH


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
