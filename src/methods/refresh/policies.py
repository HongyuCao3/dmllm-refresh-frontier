"""Refresh policies: at each denoising step, decide whether to recompute the full-sequence KV cache.

A policy sees the step index within the current block (`i`) and the global step index across blocks (`t`).
The loop always refreshes at i == 0, because a new block changes which prefix the cache must cover.
"""


class RefreshPolicy:
    name = "base"

    def reset(self):
        pass

    def __call__(self, t, i):
        raise NotImplementedError


class FixedInterval(RefreshPolicy):
    """Fast-dLLM: refresh every k steps within a block (upstream `prefix_refresh_interval`)."""

    def __init__(self, k):
        self.k = k
        self.name = f"interval{k}"

    def __call__(self, t, i):
        return i % self.k == 0


class Schedule(RefreshPolicy):
    """Refresh at an explicit set of global steps (static schedules, oracle)."""

    def __init__(self, steps, name="schedule"):
        self.steps = frozenset(steps)
        self.name = name

    def __call__(self, t, i):
        return t in self.steps
