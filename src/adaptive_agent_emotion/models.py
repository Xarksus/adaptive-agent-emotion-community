from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class AffectState:
    oxytocin: float = 20.0
    dopamine: float = 50.0
    cortisol: float = 20.0
    serotonin: float = 60.0
    noradrenaline: float = 40.0

    def as_dict(self) -> dict[str, float]:
        return {
            "oxytocin": float(self.oxytocin),
            "dopamine": float(self.dopamine),
            "cortisol": float(self.cortisol),
            "serotonin": float(self.serotonin),
            "noradrenaline": float(self.noradrenaline),
        }


@dataclass(slots=True)
class AffectEvent:
    name: str
    deltas: dict[str, float]
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
