"""Commit (unmasking) policies: at each denoising step, decide which masked positions of the current block to commit.

`select` is a copy of upstream `FastDLLMGenerationHook._get_transfer_index`
(third_party/LLaDA-V/train/llava/hooks/fast_dllm_hook.py) with two changes: the ranking score that picks the
committed positions comes from `policy.scores(...)` instead of being the confidence itself, and token sampling
(temperature > 0) draws its Gumbel noise from a seeded generator. How many positions are committed is unchanged:
the fixed schedule `n_tok` (D-order: equal NFE by construction), or upstream's threshold rule on the sorted
confidences (D-count). With `Confidence()` and temperature 0, `select` must equal upstream exactly
(tests/test_commit_select.py; gate A on the GPU: src/scripts/ga_identity.py).

A policy sees the per-position confidence `x0_p` [1, L] (probability of the predicted token, -inf where not
selectable) and the block-local positions; it returns a score [1, L], higher = committed first.
"""
import numpy as np
import torch
import torch.nn.functional as F


class CommitPolicy:
    name = "base"

    def reset(self, seed=None):
        pass

    def scores(self, x0_p, positions):
        raise NotImplementedError


class Confidence(CommitPolicy):
    """LLaDA low-confidence remasking: commit the most confident positions (upstream's rule)."""

    name = "confidence"

    def scores(self, x0_p, positions):
        return x0_p


class LeftToRight(CommitPolicy):
    """Commit the leftmost masked positions of the block (SAS Table 11: ties or beats SAS at short lengths)."""

    name = "l2r"

    def scores(self, x0_p, positions):
        return -positions.to(x0_p.dtype).expand_as(x0_p)


class _Seeded(CommitPolicy):
    """Base for policies with their own random stream; `reset(seed)` re-seeds per sample."""

    def __init__(self, seed=0):
        self.seed = seed
        self.gen = None

    def reset(self, seed=None):
        self.gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        self.gen.manual_seed(self.seed if seed is None else seed)

    def _rand(self, like):
        return torch.rand(like.shape, generator=self.gen, device=self.gen.device, dtype=torch.float64).to(like.device)


class Random(_Seeded):
    """Uniformly random commit order at the same count (the same-NFE random control)."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.name = f"random_s{seed}"

    def scores(self, x0_p, positions):
        return self._rand(x0_p)


class PerturbedConfidence(_Seeded):
    """Confidence order with Gumbel noise of scale `sigma` on log-confidence (phase B: K distinct orders)."""

    def __init__(self, sigma, seed=0):
        super().__init__(seed)
        self.sigma = sigma
        self.name = f"perturbed{sigma:g}_s{seed}"

    def scores(self, x0_p, positions):
        gumbel = -torch.log(-torch.log(self._rand(x0_p).clamp_min(1e-20)))
        return torch.log(x0_p.clamp_min(1e-300)) + self.sigma * gumbel


def build(spec):
    """Policy from a campaign arm's "commit" entry, e.g. {"policy": "perturbed", "sigma": 0.5, "seed": 3}."""
    kind = spec["policy"]
    if kind == "confidence":
        return Confidence()
    if kind == "l2r":
        return LeftToRight()
    if kind == "random":
        return Random(spec.get("seed", 0))
    if kind == "perturbed":
        return PerturbedConfidence(spec["sigma"], spec.get("seed", 0))
    raise ValueError(f"unknown commit policy {kind!r}")


def _gumbel_sample(logits, temperature, gen):
    """Upstream `_add_gumbel_noise` with a seeded generator (exp(logits) / (-log u) ** temperature, in float64)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    u = torch.rand(logits.shape, generator=gen, device=gen.device, dtype=torch.float64).to(logits.device)
    return logits.exp() / (-torch.log(u)) ** temperature


def select(policy, logits, mask_index, x, num_transfer_tokens, found_stop_seq, stop_position, block_end,
           threshold=None, temperature=0.0, gen=None, offset=0):
    """Returns (x0, transfer_index) like upstream; `offset` = absolute position of logits[:, 0]."""
    x0 = torch.argmax(_gumbel_sample(logits, temperature, gen), dim=-1)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    if found_stop_seq:
        x0_p[:, stop_position:] = -np.inf
    else:
        x0_p[:, block_end:] = -np.inf

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    positions = torch.arange(offset, offset + logits.shape[1], device=logits.device).unsqueeze(0)
    score = torch.where(torch.isfinite(confidence), policy.scores(confidence, positions), -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        if threshold is None:
            top_i = num_transfer_tokens[j]
        else:
            # Upstream's count rule, always on the confidences: the policy decides which, not how many.
            ns = list(range(1, num_transfer_tokens[j] + 1))
            threshs = [1 - threshold / (n + 1) for n in ns]
            threshs[0] = -1
            sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
            assert len(sorted_confidence) == len(threshs)
            for top_i in range(len(threshs)):
                if sorted_confidence[top_i] < threshs[top_i]:
                    break
            if top_i == 0 or top_i == len(threshs) - 1:
                top_i += 1

        _, select_index = torch.topk(score[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index
