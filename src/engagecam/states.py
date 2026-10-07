"""The 7 engagement and affect states: the single source of truth for the label order.

Also the source label rules: how a DAiSEE clip (four 0-3 affect levels) and a YawDD segment become
one state. The earlier prototype did not keep its mapping, so this rule is explicit and tested.
"""

from __future__ import annotations

STATES: tuple[str, ...] = ("active", "boredom", "confusion", "engagement", "frustration", "sleep", "yawn")
INDEX = {s: i for i, s in enumerate(STATES)}
ALIASES = {"engaged": "engagement", "bored": "boredom", "confused": "confusion", "frustrated": "frustration",
           "yawning": "yawn", "sleeping": "sleep", "asleep": "sleep", "normal": "active", "talking": "active"}

# Contribution of each state to the engagement index (0 = not engaged, 1 = fully engaged).
ENGAGEMENT_WEIGHTS = {"engagement": 1.0, "active": 0.8, "confusion": 0.5, "frustration": 0.3, "boredom": 0.2,
                      "yawn": 0.1, "sleep": 0.0}


class UnknownState(KeyError):
    """A label has no place in the state list."""


def normalise_state(raw: str) -> str:
    key = str(raw).strip().lower()
    key = ALIASES.get(key, key)
    if key not in INDEX:
        raise UnknownState(f"unknown state {raw!r}. Known: {STATES}")
    return key


def encode(labels) -> list[int]:
    return [INDEX[normalise_state(x)] for x in labels]


def daisee_state(boredom: int, engagement: int, confusion: int, frustration: int, min_level: int = 2) -> str:
    """One state for a DAiSEE clip from its four affect levels (0 = very low ... 3 = very high).

    Rule: the strongest negative affect (boredom, confusion, frustration) at ``min_level`` or more wins,
    with ties in that order. Else ``engagement`` if its level is ``min_level`` or more. Else ``active``.
    """
    levels = {"boredom": boredom, "confusion": confusion, "frustration": frustration}
    for name, value in levels.items():
        if not 0 <= int(value) <= 3:
            raise ValueError(f"{name} level {value} is not between 0 and 3")
    best = max(levels, key=lambda k: (levels[k], -list(levels).index(k)))
    if levels[best] >= min_level:
        return best
    return "engagement" if engagement >= min_level else "active"


def yawdd_state(segment: str) -> str:
    """YawDD segment names: ``normal`` and ``talking`` become ``active``, ``yawning`` becomes ``yawn``."""
    return normalise_state(segment)
