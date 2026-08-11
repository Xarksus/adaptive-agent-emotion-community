# Adaptive Agent Emotion Community

A model-agnostic affect/state engine for Python AI agents.

## Community features

- five-channel affect/state model
- configurable baselines and ceilings
- decay and recovery
- circadian baseline bias
- session-fatigue signal
- generic familiar/unfamiliar voice events
- tone-response events
- persistent JSON state
- model-neutral `policy_vector()` output
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
print(engine.policy_vector())
```

`policy_vector()` returns normalized `arousal`, `stress`, `social_affinity`, `stability`, and `fatigue` signals that can be consumed by any agent policy or prompt layer.

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

GitHub Actions tests Python 3.10, 3.11, 3.12 and 3.13 and performs a source privacy scan.

## License

MIT. See [LICENSE](LICENSE).
