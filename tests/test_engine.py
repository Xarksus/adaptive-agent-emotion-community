import json

import pytest

from adaptive_agent_emotion import AffectEvent, EmotionStateEngine


def test_inject_clamps_and_rejects_unknown():
    engine = EmotionStateEngine()
    engine.inject("cortisol", 10000)
    assert engine.state["cortisol"] == engine.ceilings["cortisol"]
    with pytest.raises(KeyError):
        engine.inject("unknown", 1)


def test_event_and_policy_vector_are_bounded():
    engine = EmotionStateEngine()
    engine.apply_event(AffectEvent("stress", {"cortisol": 50, "noradrenaline": 20}))
    policy = engine.policy_vector()
    assert set(policy) == {"arousal", "stress", "social_affinity", "stability", "fatigue"}
    assert all(0.0 <= value <= 1.0 for value in policy.values())


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "state.json"
    engine = EmotionStateEngine(path)
    engine.inject("dopamine", 10)
    engine.familiarity = 77
    engine.save()
    loaded = EmotionStateEngine(path)
    assert loaded.state["dopamine"] == pytest.approx(engine.state["dopamine"])
    assert loaded.familiarity == pytest.approx(77)
    assert json.loads(path.read_text())["state"]["dopamine"] > 0


def test_decay_moves_toward_baseline_without_negative_values():
    now = [0.0]
    engine = EmotionStateEngine(clock=lambda: now[0])
    engine.state["cortisol"] = 100
    now[0] = 3600
    engine.apply_decay()
    assert 0 <= engine.state["cortisol"] < 100


def test_voice_tone_is_generic_and_safe():
    engine = EmotionStateEngine()
    engine.on_familiar_voice(0.8)
    engine.on_unfamiliar_voice(0.1)
    engine.on_voice_tone(0.3, 0.7)
    assert all(value >= 0 for value in engine.state.values())
