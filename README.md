# Adaptive Agent Emotion (Community Edition)

![PyPI](https://img.shields.io/pypi/v/adaptive-agent-emotion)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://github.com/placeholder/adaptive-agent-emotion/actions/workflows/ci.yml/badge.svg)

## The Problem

LLM agents inherently lack internal state between interactions. Each API call to a model is completely stateless, processing inputs in isolation. While conversation history provides semantic context, it fails to capture the continuous, affective state of an agent—its "mood" or internal disposition. This absence of internal state leads to inconsistent, erratic, and unnatural behavior, especially in long-running or autonomous agents that should theoretically adapt their personality based on past interactions, failures, or successes.

## The Solution

**Adaptive Agent Emotion** introduces a persistent, biologically-inspired affective state model for autonomous agents. By maintaining an internal simulated neuromodulator state, your agents can react, adapt, and modulate their behavior continuously.

### Installation

```bash
pip install adaptive-agent-emotion
```

### The 5-Channel Neuromodulatory Model

Our model is driven by five core channels, mapped to synthetic neurotransmitter analogs:

- **Dopamine**: Represents reward and motivation. Drives the agent to pursue goals and increases focus.
- **Serotonin**: Represents satisfaction and social well-being. Promotes stability and cooperative behavior.
- **Cortisol**: Represents stress and urgency. Elevates alertness but can lead to erratic behavior if sustained.
- **Noradrenaline**: Represents arousal and vigilance. Controls the agent's immediate readiness to react to stimuli.
- **Oxytocin**: Represents social affinity and trust. Influences the agent's willingness to collaborate and bond.

### How It Works

- **Decay**: Emotions aren't permanent. Over time, neuromodulator levels decay toward their baseline state, simulating emotional recovery and habituation.
- **Circadian Bias**: The system can apply temporal biases, mimicking how circadian rhythms naturally affect baseline arousal, stress, and fatigue throughout a simulated day.
- **Cross-Coupling**: Channels interact with each other. For instance, high sustained Cortisol (stress) might gradually suppress Serotonin (well-being), creating complex, emergent emotional states rather than isolated sliders.

### Generating the Policy Vector

The internal state isn't just for show; it directly informs agent behavior through the `policy_vector()`. This vector translates the raw neuromodulator levels into five actionable behavioral metrics:

- **Arousal**: How active, verbose, or proactive the agent should be.
- **Stress**: How urgently or defensively the agent should respond.
- **Social Affinity**: How friendly, cooperative, or trusting the agent acts.
- **Stability**: The predictability and consistency of the agent's choices.
- **Fatigue**: The likelihood of the agent making errors, giving shorter responses, or requiring "rest".

### Quick Start Example

Here's how to integrate affective states into your agent's event loop:

```python
from adaptive_agent_emotion import EmotionEngine, AffectEvent

# Initialize the engine
engine = EmotionEngine()

# Apply an event (e.g., the agent received a harsh user prompt)
engine.apply_event(AffectEvent(
    name="harsh_criticism",
    cortisol_delta=0.4,
    serotonin_delta=-0.2,
    oxytocin_delta=-0.3
))

# Retrieve the behavioral policy vector to modulate your LLM prompt
policy = engine.policy_vector()

print(f"Current Arousal: {policy.arousal}")
print(f"Current Stress: {policy.stress}")
print(f"Current Social Affinity: {policy.social_affinity}")

# Use the policy vector to adjust your agent's system prompt or generation parameters
```

### Community vs. Pro

| Feature | Community Edition | Pro Edition |
|---------|-------------------|-------------|
| 5-Channel Neuromodulator Model | ✅ | ✅ |
| AffectEvent Processing | ✅ | ✅ |
| Basic Decay Mechanics | ✅ | ✅ |
| Policy Vector Generation | ✅ | ✅ |
| Advanced Circadian Rhythms | ❌ | ✅ |
| Complex Cross-Coupling | ❌ | ✅ |
| Memory Persistence (DB Integration)| ❌ | ✅ |
| Multi-Agent Social Graphs | ❌ | ✅ |

### License

This project is licensed under the MIT License - see the LICENSE file for details.
