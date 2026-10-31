"""CPU check of src/methods/commit/policies.select against upstream `_get_transfer_index`.

Runs in the llada-v env (needs `llava`): `python -m pytest tests/test_commit_select.py`.
"""
import pytest
import torch

pytest.importorskip("llava")
from llava.hooks.fast_dllm_hook import FastDLLMGenerationHook  # noqa: E402

from src.methods.commit.policies import Confidence, LeftToRight, PerturbedConfidence, Random, select  # noqa: E402

V, L = 64, 48


def case(seed, n_masked, block_end, found_stop=False, stop=40):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, L, V, generator=g) * 3
    mask = torch.zeros(1, L, dtype=torch.bool)
    mask[0, torch.randperm(block_end, generator=g)[:n_masked]] = True
    x = torch.randint(0, V, (1, L), generator=g)
    return logits, mask, x, found_stop, stop, block_end


CASES = [case(s, n, b, fs) for s in range(6) for n, b in ((32, 32), (10, 32), (1, 32), (20, 40))
         for fs in (False, True)]


@pytest.mark.parametrize("threshold", [None, 0.5, 0.9])
@pytest.mark.parametrize("c", CASES)
def test_confidence_equals_upstream(c, threshold):
    logits, mask, x, fs, stop, bend = c
    n_tok = torch.tensor([[min(4, int(mask.sum()))]])[:, 0]
    up = object.__new__(FastDLLMGenerationHook)
    x0_u, ti_u = up._get_transfer_index(logits, 0.0, "low_confidence", mask, x, n_tok, fs, stop, bend, 0,
                                        threshold)
    x0_o, ti_o = select(Confidence(), logits, mask, x, n_tok, fs, stop, bend, threshold)
    assert torch.equal(x0_u, x0_o)
    assert torch.equal(ti_u, ti_o)


@pytest.mark.parametrize("policy", [LeftToRight(), Random(1), PerturbedConfidence(1.0, 2)])
@pytest.mark.parametrize("c", CASES[:8])
def test_other_policies_keep_count_and_eligibility(c, policy):
    logits, mask, x, fs, stop, bend = c
    n_tok = torch.tensor([[min(4, int(mask.sum()))]])[:, 0]
    policy.reset(0)
    _, ti_ref = select(Confidence(), logits, mask, x, n_tok, fs, stop, bend)
    _, ti = select(policy, logits, mask, x, n_tok, fs, stop, bend)
    limit = stop if fs else bend
    assert ti.sum() == ti_ref.sum()
    assert not (ti & ~mask).any()
    assert not ti[0, limit:].any()


def test_left_to_right_picks_leftmost():
    logits, mask, x, fs, stop, bend = case(0, 10, 32)
    n_tok = torch.tensor([3])
    _, ti = select(LeftToRight(), logits, mask, x, n_tok, False, stop, bend)
    assert ti[0].nonzero().flatten().tolist() == mask[0].nonzero().flatten()[:3].tolist()


def test_seeded_policies_repeat():
    logits, mask, x, fs, stop, bend = case(3, 20, 32)
    n_tok = torch.tensor([4])
    outs = []
    for _ in range(2):
        p = Random(7)
        p.reset()
        outs.append(select(p, logits, mask, x, n_tok, False, stop, bend)[1])
    assert torch.equal(*outs)


def test_temperature_sampling_seeded_and_differs():
    logits, mask, x, fs, stop, bend = case(4, 32, 32)
    n_tok = torch.tensor([4])
    a = select(Confidence(), logits, mask, x, n_tok, False, stop, bend, None, 0.5, torch.Generator().manual_seed(1))
    b = select(Confidence(), logits, mask, x, n_tok, False, stop, bend, None, 0.5, torch.Generator().manual_seed(1))
    c = select(Confidence(), logits, mask, x, n_tok, False, stop, bend, None, 0.0)
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_recorder_tracks_flips_and_commits():
    from src.methods.commit.signals import Recorder
    rec = Recorder(8)
    V = 5
    big = lambda tok: torch.nn.functional.one_hot(torch.tensor(tok), V).float() * 5
    masked = torch.ones(4, dtype=torch.bool)
    # step 0: argmax [1,1,1,1], commit position 0
    rec.observe(0, 4, big([1, 1, 1, 1]), masked, torch.tensor([True, False, False, False]), torch.tensor([1, 1, 1, 1]))
    masked = torch.tensor([False, True, True, True])
    # step 1: position 1 flips to 2 and is committed; position 0 is no longer masked (no flip counted)
    rec.observe(1, 4, big([3, 2, 1, 1]), masked, torch.tensor([False, True, False, False]), torch.tensor([1, 2, 1, 1]))
    j = rec.to_json(8)
    assert j["first_argmax"][4:] == [1, 1, 1, 1] and j["first_argmax"][:4] == [-1] * 4
    assert j["flips"][4:] == [0, 1, 0, 0]
    assert j["commit_step"][4:6] == [0, 1] and j["commit_step"][6] == -1
    assert j["commit_logp"][4] > -0.1 and j["commit_margin"][5] > 4
