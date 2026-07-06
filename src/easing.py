from __future__ import annotations

import math


EASING_CHOICES = ("linear", "cosine", "logarithmic")
LOGARITHMIC_EASING_STRENGTH = 12.0


def ease_progress(t: float, easing: str) -> float:
    if easing == "linear":
        return t
    if easing == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if easing == "logarithmic":
        return math.log1p(LOGARITHMIC_EASING_STRENGTH * t) / math.log1p(
            LOGARITHMIC_EASING_STRENGTH
        )
    raise ValueError(f"Unsupported easing: {easing}")
