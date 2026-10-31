"""Per-position decoding signals for the reward-signal exploration (plan v3, phase B2). No extra forward passes:
everything is read from the logits the loop already computes.

For every answer position the recorder keeps the argmax at the first step the position was visible in its
block, how often the argmax changed while the position stayed masked, the step it was committed at, and at that
step the log-probability of the committed token and the top-1 - top-2 log-probability margin. Trajectory-level
signals (path log-likelihood = SAS's reward without a reference text, low-margin commit count, flips) are
derived from these arrays in the analysis, not here. All values are diagnostics; none is a reported quality
number (`.claude/global.md`).
"""
import torch


class Recorder:
    def __init__(self, gen_length):
        n = gen_length
        self.first_argmax = torch.full((n,), -1, dtype=torch.long)
        self.last_argmax = torch.full((n,), -1, dtype=torch.long)
        self.flips = torch.zeros(n, dtype=torch.long)
        self.commit_step = torch.full((n,), -1, dtype=torch.long)
        self.commit_logp = torch.full((n,), float("nan"))
        self.commit_margin = torch.full((n,), float("nan"))

    @torch.no_grad()
    def observe(self, t, start, logits, masked, committed, tokens):
        """One denoising step on one block.

        t: global step; start: block offset within the answer; logits [B, V] for the block's positions;
        masked [B] positions still masked before this step; committed [B] positions committed at this step;
        tokens [B] the token at each position after this step.
        """
        lp = torch.log_softmax(logits.float(), dim=-1)
        top2 = lp.topk(2, dim=-1)
        am = top2.indices[:, 0].cpu()
        masked, committed = masked.cpu(), committed.cpu()
        idx = torch.arange(start, start + len(am))
        seen = masked & (self.first_argmax[idx] >= 0)
        self.flips[idx[seen]] += (am[seen] != self.last_argmax[idx[seen]]).long()
        new = masked & (self.first_argmax[idx] < 0)
        self.first_argmax[idx[new]] = am[new]
        self.last_argmax[idx[masked]] = am[masked]
        if committed.any():
            c = committed.nonzero().flatten()
            tok = tokens[c.to(tokens.device)]
            self.commit_step[idx[c]] = t
            self.commit_logp[idx[c]] = lp[c.to(lp.device), tok].float().cpu()
            self.commit_margin[idx[c]] = (top2.values[c.to(lp.device), 0] - top2.values[c.to(lp.device), 1]).float().cpu()

    def to_json(self, n):
        """Arrays truncated to the n answer tokens actually returned."""
        r = lambda v: [round(float(a), 4) for a in v[:n]]
        return dict(first_argmax=self.first_argmax[:n].tolist(), flips=self.flips[:n].tolist(),
                    commit_step=self.commit_step[:n].tolist(), commit_logp=r(self.commit_logp),
                    commit_margin=r(self.commit_margin))
