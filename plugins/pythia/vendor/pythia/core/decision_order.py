"""Order-preserving closure of simultaneous decision-risk certificates."""
from __future__ import annotations

import math
from collections.abc import Mapping


def monotone_window_floor_closure(bounds: Mapping[float, float]) -> dict[float, float]:
    """Tighten simultaneous miss-risk lower bounds across increasing windows.

    Bayes miss risk is nonincreasing in window width. Therefore a valid lower
    bound at any larger width is also valid at every smaller width. Input
    bounds must already be jointly valid, e.g. by an outer alpha allocation;
    this deterministic closure spends no additional alpha and cannot repair
    outcome-adaptive selection or a false smoothness cap.
    """
    if not bounds:
        raise ValueError("a nonempty predeclared window registry is required")
    rows = sorted((float(width), float(bound)) for width, bound in bounds.items())
    if any(not math.isfinite(width) or width <= 0
           or not math.isfinite(bound) or not 0 <= bound <= 1
           for width, bound in rows):
        raise ValueError("finite positive windows and miss-risk bounds in [0,1] required")
    if len({width for width, _ in rows}) != len(rows):
        raise ValueError("window widths must be unique")
    running = 0.0
    closed: dict[float, float] = {}
    for width, bound in reversed(rows):
        running = max(running, bound)
        closed[width] = running
    return {width: closed[width] for width, _ in rows}
