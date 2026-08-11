import json
import pytest
from adaptive_agent_emotion import AffectEvent, EmotionStateEngine


def test_inject_clamps_and_rejects_unknown():
    engine=EmotionStateEngine(); engine.inject("cortisol",10000); assert engine.state["cortisol"]==engine.ceilings["cortisol"]
    with pytest.raises(KeyError): engine.inject("unknown",1)

def test_event_and_policy_vector_are_bounded_and_pure():
    now=[0.0]; engine=EmotionStateEngine(clock=lambda:now[0]); engine.apply_event(AffectEvent("stress",{"cortisol":50,"noradrenaline":20})); before=dict(engine.state); p1=engine.policy_vector(); now[0]=180; p2=engine.policy_vector(); assert engine.state==before; assert p1["stress"]==pytest.approx(p2["stress"]); assert all(0<=v<=1 for v in p1.values())

def test_persistence_round_trip(tmp_path):
    path=tmp_path/"state.json"; engine=EmotionStateEngine(path); engine.inject("dopamine",10); engine.familiarity=77; engine.on_voice_tone(.4,.2); engine.save(); loaded=EmotionStateEngine(path); assert loaded.state["dopamine"]==pytest.approx(engine.state["dopamine"],abs=.05); assert loaded._voice_norm==pytest.approx(engine._voice_norm); assert json.loads(path.read_text())["state"]["dopamine"]>0

def test_load_does_not_write_back(tmp_path):
    path=tmp_path/"state.json"; original={"state":{"oxytocin":20,"dopamine":100,"cortisol":100,"serotonin":60,"noradrenaline":40},"baselines":{"oxytocin":20,"dopamine":50,"cortisol":20,"serotonin":60,"noradrenaline":40},"familiarity":50,"voice_norm":0,"last_update":0}; path.write_text(json.dumps(original,indent=2)); before=path.read_text(); EmotionStateEngine(path,clock=lambda:3600.0); assert path.read_text()==before

def test_decay_moves_toward_effective_baseline_without_negative_values():
    now=[0.0]; engine=EmotionStateEngine(clock=lambda:now[0]); engine.state["cortisol"]=100; now[0]=3600; engine.apply_decay(hour=12); assert 0<=engine.state["cortisol"]<100

def test_circadian_bias_never_accumulates_into_baselines():
    engine=EmotionStateEngine(); original=dict(engine.baselines); first=engine.apply_circadian_bias(18)
    for _ in range(100): assert engine.apply_circadian_bias(18)==first
    assert engine.baselines==original

def test_tick_is_explicit_mutation_boundary():
    now=[0.0]; engine=EmotionStateEngine(clock=lambda:now[0]); engine.state["cortisol"]=100; now[0]=60; engine.tick(hour=12); assert engine.state["cortisol"]<100

def test_session_reset():
    now=[0.0]; engine=EmotionStateEngine(clock=lambda:now[0]); now[0]=5*3600; assert engine.policy_vector()["fatigue"]==pytest.approx(1); engine.reset_session(); assert engine.policy_vector()["fatigue"]==pytest.approx(0)

def test_familiarity_bounded():
    engine=EmotionStateEngine()
    for _ in range(500): engine.on_familiar_voice(1)
    assert engine.familiarity==100

def test_serotonin_ceiling(): assert EmotionStateEngine().ceilings["serotonin"]==300

def test_rest_recovery_converges_instead_of_ratchets():
    now=[0.0]; e=EmotionStateEngine(clock=lambda:now[0]); e.baselines.update({"dopamine":50,"serotonin":60,"cortisol":20})
    now[0]=12*3600; e.apply_decay(hour=12); first=dict(e.baselines)
    for day in range(1,20): now[0]+=12*3600; e.apply_decay(hour=12)
    assert e.baselines["dopamine"]==pytest.approx(58); assert e.baselines["serotonin"]==pytest.approx(66); assert e.baselines["cortisol"]==pytest.approx(15); assert e.baselines==first

def test_dominant_signals_use_effective_baseline():
    e=EmotionStateEngine(); e.state["serotonin"]=68; assert e.dominant_signals(hour=18)["serotonin"]==pytest.approx(0)

def test_voice_reference_resists_habituation_to_persistent_loudness():
    e=EmotionStateEngine(); e.on_voice_tone(1.0,0); initial=e._voice_norm
    hard=0
    for _ in range(100):
        before=len(e.events); e.on_voice_tone(2.0,1.0); hard += any(x.name=="voice_tone_hard" for x in e.events[before:])
    assert hard>=90; assert e._voice_norm<1.25; assert e._voice_norm>initial

def test_voice_reference_can_recover_downward():
    e=EmotionStateEngine(); e.on_voice_tone(2.0,0); start=e._voice_norm
    for _ in range(100): e.on_voice_tone(.5,0)
    assert e._voice_norm<start

def test_corrupt_state_raises_clear_error(tmp_path):
    p=tmp_path/"state.json"; p.write_text("{broken")
    with pytest.raises(ValueError,match="Could not load emotion state"): EmotionStateEngine(p)
