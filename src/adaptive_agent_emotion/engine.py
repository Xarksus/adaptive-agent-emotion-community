from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .models import AffectEvent, AffectState

HORMONES = ("oxytocin", "dopamine", "cortisol", "serotonin", "noradrenaline")
DEFAULT_CEILINGS = {"oxytocin": 400.0, "dopamine": 300.0, "cortisol": 300.0, "serotonin": 300.0, "noradrenaline": 400.0}
DEFAULT_BASELINE_CEILINGS = {"oxytocin": 250.0, "dopamine": 150.0, "cortisol": 120.0, "serotonin": 150.0, "noradrenaline": 120.0}
DEFAULT_DECAY_RATES = {"oxytocin": 0.25, "dopamine": 0.50, "cortisol": 0.80, "serotonin": 0.20, "noradrenaline": 0.70}
RESTED_BASELINES = {"dopamine": 58.0, "serotonin": 66.0, "cortisol": 15.0}
VOICE_NORM_UP_ALPHA = 0.002
VOICE_NORM_DOWN_ALPHA = 0.05


def _circadian_offsets(hour: int) -> dict[str, float]:
    hour %= 24
    if 6 <= hour < 9: return {"cortisol": 10.0, "noradrenaline": 8.0, "dopamine": 3.0}
    if 10 <= hour < 13: return {"dopamine": 10.0, "serotonin": 5.0, "cortisol": -3.0}
    if 14 <= hour < 16: return {"dopamine": -8.0, "noradrenaline": -5.0}
    if 17 <= hour < 21: return {"serotonin": 8.0, "oxytocin": 3.0, "cortisol": -6.0}
    return {"dopamine": -10.0, "noradrenaline": -10.0, "cortisol": -8.0}


class EmotionStateEngine:
    """Model-agnostic affect/state engine. Time mutation is explicit via tick()."""
    def __init__(self, state_path: str | Path | None = None, *, baselines: AffectState | None = None, ceilings: dict[str, float] | None = None, baseline_ceilings: dict[str, float] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.state_path = Path(state_path).expanduser() if state_path else None
        self.clock = clock
        self.baselines = (baselines or AffectState()).as_dict(); self.state = dict(self.baselines)
        self.ceilings = dict(DEFAULT_CEILINGS if ceilings is None else ceilings); self.baseline_ceilings = dict(DEFAULT_BASELINE_CEILINGS if baseline_ceilings is None else baseline_ceilings)
        self.last_update = self.clock(); self._last_coupling = self.last_update; self.session_start = self.last_update
        self.familiarity = 50.0; self._voice_norm = 0.0; self.events: list[AffectEvent] = []
        if self.state_path and self.state_path.exists(): self.load()

    def snapshot(self) -> AffectState:
        return AffectState(**{name: self.state[name] for name in HORMONES})

    def load(self) -> None:
        if not self.state_path: return
        try: data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc: raise ValueError(f"Could not load emotion state from {self.state_path}") from exc
        for name in HORMONES:
            if name in data.get("state", {}): self.state[name] = float(data["state"][name])
            if name in data.get("baselines", {}): self.baselines[name] = float(data["baselines"][name])
        self.familiarity = float(data.get("familiarity", self.familiarity)); self.last_update = float(data.get("last_update", self.clock())); self._voice_norm = float(data.get("voice_norm", self._voice_norm))
        self._clamp(); self.apply_decay()

    def save(self) -> None:
        if not self.state_path: return
        self.state_path.parent.mkdir(parents=True, exist_ok=True); self._clamp()
        text = json.dumps({"state": self.state, "baselines": self.baselines, "familiarity": self.familiarity, "voice_norm": self._voice_norm, "last_update": self.last_update}, indent=2)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle: handle.write(text); handle.flush(); os.fsync(handle.fileno())
            os.replace(tmp_name, self.state_path)
        except Exception:
            try: os.unlink(tmp_name)
            except OSError: pass
            raise

    def _clamp(self) -> None:
        for name in HORMONES:
            self.state[name] = max(0.0, min(float(self.state[name]), self.ceilings[name])); self.baselines[name] = max(0.0, min(float(self.baselines[name]), self.baseline_ceilings[name]))
        self.familiarity = max(0.0, min(float(self.familiarity), 100.0))

    def effective_baselines(self, hour: int | None = None) -> dict[str, float]:
        hour = datetime.now(timezone.utc).hour if hour is None else int(hour) % 24; offsets = _circadian_offsets(hour)
        return {name: max(0.0, min(self.baselines[name] + offsets.get(name, 0.0), self.baseline_ceilings[name])) for name in HORMONES}

    def apply_circadian_bias(self, hour: int | None = None) -> dict[str, float]: return self.effective_baselines(hour)

    def _apply_rest_recovery(self, hours: float) -> None:
        """Converge baseline recovery toward rested targets; never ratchet past them."""
        if hours < 6.0: return
        recovery = min((hours - 6.0) / 6.0, 1.0)
        for name, target in RESTED_BASELINES.items(): self.baselines[name] += (target - self.baselines[name]) * recovery

    def apply_decay(self, *, now: float | None = None, hour: int | None = None) -> None:
        now = self.clock() if now is None else float(now); hours = (now - self.last_update) / 3600.0
        if hours <= 0: return
        self._apply_rest_recovery(hours); self._clamp(); targets = self.effective_baselines(hour)
        for name, rate in DEFAULT_DECAY_RATES.items(): self.state[name] += (targets[name] - self.state[name]) * min(1.0, hours * rate)
        self.familiarity = max(0.0, self.familiarity - hours); self.last_update = now; self._clamp()

    def tick(self, *, now: float | None = None, hour: int | None = None) -> None:
        tick_now = self.clock() if now is None else float(now); self.apply_decay(now=tick_now, hour=hour); self.apply_cross_coupling(now=tick_now)

    def inject(self, hormone: str, amount: float, *, context: str = "") -> None:
        if hormone not in HORMONES: raise KeyError(f"Unknown hormone/state channel: {hormone}")
        self.state[hormone] += float(amount); self._clamp(); self.events.append(AffectEvent(name=f"inject:{hormone}", deltas={hormone: amount}, context=context))

    def apply_event(self, event: AffectEvent) -> None:
        for hormone, delta in event.deltas.items():
            if hormone not in HORMONES: raise KeyError(f"Unknown hormone/state channel: {hormone}")
            self.state[hormone] += float(delta)
        self._clamp(); self.events.append(event)

    def apply_cross_coupling(self, *, now: float | None = None) -> None:
        now = self.clock() if now is None else float(now); dt = now - self._last_coupling; self._last_coupling = now
        if dt <= 0: return
        rate = min(dt, 60.0); oxy, ser, cor = self.state["oxytocin"], self.state["serotonin"], self.state["cortisol"]
        buffer = (oxy / 100.0) * 8.0 + max(0.0, (ser - 50.0) / 50.0) * 5.0; self.state["cortisol"] = max(0.0, cor - buffer * 0.02 * rate)
        if self.state["cortisol"] > 80:
            suppression = (self.state["cortisol"] - 80.0) / 20.0 * 3.0; self.state["serotonin"] = max(0.0, self.state["serotonin"] - suppression * 0.02 * rate)
        fatigue = self.session_fatigue(now=now)
        if fatigue > 60:
            drag = (fatigue - 60.0) * 0.005 * rate; self.state["dopamine"] = max(0.0, self.state["dopamine"] - drag); self.state["noradrenaline"] = max(0.0, self.state["noradrenaline"] - drag * 0.5)
        self._clamp()

    def session_fatigue(self, *, now: float | None = None) -> float:
        now = self.clock() if now is None else float(now); return max(0.0, (now - self.session_start) / 60.0) * 0.6
    def reset_session(self, *, now: float | None = None) -> None: self.session_start = self.clock() if now is None else float(now)
    def on_conversation_end(self, *, abrupt: bool = False, context: str = "") -> None:
        event = AffectEvent("conversation_abrupt_end", {"oxytocin": -8.0, "cortisol": 6.0, "serotonin": -3.0}, context) if abrupt else AffectEvent("conversation_normal_end", {"serotonin": 4.0, "dopamine": 2.0}, context)
        self.apply_event(event); self.reset_session()
    def on_familiar_voice(self, similarity: float = 1.0) -> None:
        similarity = max(0.0, min(1.0, float(similarity))); self.familiarity = min(100.0, self.familiarity + 0.5 * similarity); self.apply_event(AffectEvent("familiar_voice", {"oxytocin": 2.5 * similarity, "serotonin": 1.5 * similarity}, metadata={"similarity": similarity}))
    def on_unfamiliar_voice(self, similarity: float = 0.0) -> None:
        similarity = max(0.0, min(1.0, float(similarity))); surprise = 1.0 - similarity; self.apply_event(AffectEvent("unfamiliar_voice", {"noradrenaline": 4.0 * surprise, "cortisol": 2.0 * surprise}, metadata={"similarity": similarity}))

    def on_voice_tone(self, loudness: float, sharpness: float = 0.0) -> None:
        loudness = max(0.0, float(loudness)); sharpness = max(0.0, min(1.0, float(sharpness)))
        if self._voice_norm <= 0: self._voice_norm = max(0.01, loudness)
        else:
            alpha = VOICE_NORM_UP_ALPHA if loudness > self._voice_norm else VOICE_NORM_DOWN_ALPHA
            self._voice_norm += (loudness - self._voice_norm) * alpha
        rel = max(-1.0, min(2.0, (loudness - self._voice_norm) / (self._voice_norm + 1e-3))); hardness = max(0.0, rel) * (0.4 + 0.6 * sharpness); warmth = max(0.0, 0.2 - rel) * (1.0 - sharpness)
        if hardness > 0.15:
            self.apply_event(AffectEvent("voice_tone_hard", {"noradrenaline": 6.0 * hardness, "cortisol": 5.0 * hardness, "serotonin": -2.5 * hardness})); self.familiarity = max(0.0, self.familiarity - 2.0 * hardness)
        elif warmth > 0.10: self.apply_event(AffectEvent("voice_tone_warm", {"oxytocin": 1.0 * warmth, "serotonin": 1.5 * warmth, "noradrenaline": -2.0 * warmth}))

    def dominant_signals(self, hour: int | None = None) -> dict[str, float]:
        targets = self.effective_baselines(hour); return {name: (self.state[name] - targets[name]) / max(1.0, targets[name]) for name in HORMONES}

    def policy_vector(self) -> dict[str, float]:
        s = self.state; hormone_affinity = min(1.0, max(0.0, s["oxytocin"] / 200.0)); familiarity_affinity = self.familiarity / 100.0
        return {"arousal": min(1.0, max(0.0, (s["noradrenaline"] + s["dopamine"]) / 300.0)), "stress": min(1.0, max(0.0, s["cortisol"] / 150.0)), "social_affinity": min(1.0, max(0.0, 0.75 * hormone_affinity + 0.25 * familiarity_affinity)), "stability": min(1.0, max(0.0, s["serotonin"] / 200.0)), "fatigue": min(1.0, self.session_fatigue() / 180.0)}
