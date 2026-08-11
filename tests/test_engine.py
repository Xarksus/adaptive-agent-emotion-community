import json

import pytest

from adaptive_agent_emotion import AffectEvent, EmotionStateEngine


def test_inject_clamps_and_rejects_unknown():
    engine = EmotionStateEngine()
    engine.inject("cortisol", 10000)
    assert engine.state["cortisol"] == engine.ceilings["cortisol"]
    with pytest.raises(KeyError):
        engine.inject("unknown", 1)


def test_event_and_policy_vector_are_bounded_and_pure():
    now = [0.0]
    engine = EmotionStateEngine(clock=lambda: now[0])
    engine.apply_event(AffectEvent("stress", {"cortisol": 50, "noradrenaline": 20}))
    before = dict(engine.state)
    policy_1 = engine.policy_vector()
    now[0] = 180.0
    policy_2 = engine.policy_vector()
    assert set(policy_1) == {"arousal", "stress", "social_affinity", "stability", "fatigue"}
    assert all(0.0 <= value <= 1.0 for value in policy_1.values())
    assert engine.state == before
    assert policy_1["stress"] == pytest.approx(policy_2["stress"])


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "state.json"
    engine = EmotionStateEngine(path)
    engine.inject("dopamine", 10)
    engine.familiarity = 77
    engine.on_voice_tone(0.4, 0.2)
    engine.save()
    loaded = EmotionStateEngine(path)
    assert loaded.state["dopamine"] == pytest.approx(engine.state["dopamine"], abs=0.05)
    assert loaded.familiarity == pytest.approx(77, abs=0.05)
    assert loaded._voice_norm == pytest.approx(engine._voice_norm)
    assert json.loads(path.read_text())["state"]["dopamine"] > 0


def test_load_does_not_write_back(tmp_path):
    path = tmp_path / "state.json"
    original = {
        "state": {"oxytocin": 20, "dopamine": 100, "cortisol": 100, "serotonin": 60, "noradrenaline": 40},
        "baselines": {"oxytocin": 20, "dopamine": 50, "cortisol": 20, "serotonin": 60, "noradrenaline": 40},
        "familiarity": 50,
        "voice_norm": 0,
        "last_update": 0,
    }
    path.write_text(json.dumps(original, indent=2))
    before = path.read_text()
    EmotionStateEngine(path, clock=lambda: 3600.0)
    assert path.read_text() == before


def test_decay_moves_toward_effective_baseline_without_negative_values():
    now = [0.0]
    engine = EmotionStateEngine(clock=lambda: now[0])
    engine.state["cortisol"] = 100
    now[0] = 3600
    engine.apply_decay(hour=12)
    assert 0 <= engine.state["cortisol"] < 100


def test_circadian_bias_never_accumulates_into_baselines():
    engine = EmotionStateEngine()
    original = dict(engine.baselines)
    first = engine.apply_circadian_bias(18)
    for _ in range(100):
        assert engine.apply_circadian_bias(18) == first
    assert engine.baselines == original
    assert first["serotonin"] == pytest.approx(original["serotonin"] + 8.0)


def test_tick_is_the_explicit_mutation_boundary():
    now = [0.0]
    engine = EmotionStateEngine(clock=lambda: now[0])
    engine.state["cortisol"] = 100
    before = engine.state["cortisol"]
    now[0] = 60.0
    engine.tick(hour=12)
    assert engine.state["cortisol"] < before


def test_session_can_be_reset_and_conversation_end_resets_it():
    now = [0.0]
    engine = EmotionStateEngine(clock=lambda: now[0])
    now[0] = 5 * 60 * 60
    assert engine.policy_vector()["fatigue"] == pytest.approx(1.0)
    engine.reset_session()
    assert engine.policy_vector()["fatigue"] == pytest.approx(0.0)
    now[0] += 3600
    engine.on_conversation_end()
    assert engine.policy_vector()["fatigue"] == pytest.approx(0.0)


def test_familiarity_is_bounded_and_affects_social_affinity():
    engine = EmotionStateEngine()
    base = engine.policy_vector()["social_affinity"]
    for _ in range(500):
        engine.on_familiar_voice(1.0)
    assert engine.familiarity == 100.0
    assert engine.policy_vector()["social_affinity"] >= base


def test_voice_tone_is_generic_and_safe():
    engine = EmotionStateEngine()
    engine.on_familiar_voice(0.8)
    engine.on_unfamiliar_voice(0.1)
    engine.on_voice_tone(0.3, 0.7)
    assert all(value >= 0 for value in engine.state.values())
    assert engine._voice_norm > 0


def test_serotonin_ceiling_matches_useful_policy_range():
    engine = EmotionStateEngine()
    assert engine.ceilings["serotonin"] == 300.0
