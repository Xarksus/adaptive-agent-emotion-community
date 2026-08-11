# Adaptive Agent Emotion Community

A model-agnostic affect/state engine for Python AI agents.

## Community features

- five-channel affect/state model
- configurable baselines and ceilings
- non-accumulating circadian baseline offsets
- explicit time advancement through `tick()`
- session-fatigue signal and reset support
- generic familiar/unfamiliar voice events
- tone-response events
- atomic JSON persistence
- read-only model-neutral `policy_vector()` output
- dependency-free Python core

## Install

```bash
pip install -e .
```

Requires Python 3.10+.

## Quick start

```python
from adaptive_agent_emotion import AffectEvent, EmotionStateEngine

engine = EmotionStateEngine("agent_state.json")
engine.apply_event(AffectEvent("tool_failed", {"cortisol": 12, "dopamine": -5}))
engine.tick()  # explicit time-dependent decay/coupling
print(engine.policy_vector())  # read-only
engine.save()
```

`policy_vector()` returns normalized `arousal`, `stress`, `social_affinity`, `stability`, and `fatigue` signals. Reading the vector does not advance or mutate the engine state.

Circadian behavior is applied as an effective baseline offset during `tick()` / `apply_decay()` and does not accumulate into stored baselines.

## Community vs Pro

This repository is the MIT-licensed Community edition. It intentionally contains the affect/state engine only.

The proprietary Pro layer adds decision-feedback learning, including expectation-vs-outcome observation, adaptive learning rates for repeated contexts, SQLite experience history, advisor summaries from prior outcomes, and advanced state-conditioned decision policies/analytics.

See [PRO.md](PRO.md) for the product boundary.

## Privacy

This Community repository is intentionally generic. It does not contain the private source persona/runtime or the proprietary observer/advisor implementation.

## Tests

```bash
pip install -e . pytest
pytest -q
```

GitHub Actions tests Python 3.10, 3.11, 3.12 and 3.13, verifies the Community/Pro boundary, and scans public source, tests and product documentation for private markers.

## License

MIT. See [LICENSE](LICENSE).
