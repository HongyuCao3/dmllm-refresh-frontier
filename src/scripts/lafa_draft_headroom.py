"""Backup line (plan v3, "备用主线"), experiment E1: how much headroom do the order and the time schedule of Lafa's
flow drafts leave for self-speculative decoding, and when is each token determined during the flow.

Lafa's self-speculative decoding (HF checkpoint `flow_model.generate_self_speculative`) drafts a 4-token block with
one flow-map jump from noise (t 0 -> 1 at every position, so the default draft has no order) and commits the longest
prefix that matches greedy AR. Its output is AR greedy, so draft quality only moves speed: tokens per forward.

Teacher-forced protocol: y = the self-speculative output (= AR greedy, checked token by token against the clean
next-token logits). At every `--stride`-th position p of y the prefix y[:p] is in the KV cache and each draft
variant proposes y[p:p+4] from the same noise. Recorded per variant and jump: argmax and top-1 probability per
slot; for `warp4` also the token nearest to the latent state (the "latent determination" of a position).

Variants (grids over base time u = k/K, per-position times tau = learned pooled warp(u + offset), endpoints pinned):
  warp1  default self-speculative draft (one jump)          warp2, warp4  learned per-position grid, more jumps
  l2r4   left positions ahead: offset c * (0.5 - i/3)        rev4  right positions ahead (offset negated)
  conf4  content order: after jump 1, offsets by confidence rank (most confident ahead), applied from column 2 on
The learned per-position grids of the released checkpoints are almost synchronous (7B, K=2 midpoint: 0.079 at
position 0 vs 0.063-0.066 at 1-3), so l2r4 / rev4 / conf4 are off the training distribution; their effect mixes
order and distribution shift. All values are diagnostics (headroom bounds), not reported quality numbers.

Before any problem is run, the `warp4` draft loop is checked against the checkpoint's own `_denoise_flowmap_block`
(same seed, same cache): mismatch aborts the job.

    python src/scripts/lafa_draft_headroom.py --task gsm8k --n 100 --shard 0 --num-shards 2
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

ACCT = os.popen("id -un").read().strip()
# Prompts as in dKV-ARKV src/dkvar/eval/tasks.py (from NLD's evaluator), so the L0 traces are comparable.
TASKS = {"gsm8k": ("openai/gsm8k", "main", "test", "question",
                   "Solve the following math problem. Put the final numerical answer inside \\boxed{} at the very end.\n\n"),
         "math-500": ("HuggingFaceH4/MATH-500", None, "test", "problem",
                      "Solve the following math problem. Put the final answer inside \\boxed{} at the very end.\n\n")}
VARIANTS = {"warp1": (1, None), "warp2": (2, None), "warp4": (4, None), "l2r4": (4, "l2r"), "rev4": (4, "rev"),
            "conf4": (4, "conf")}


class Drafter:
    def __init__(self, model, stagger):
        self.m, self.c = model, stagger
        model._ensure_time_warper(model.embed_tokens.weight.device)
        self.w = model.time_warper
        self.edges = self.w._get_bin_widths_and_edges()
        self.trim = type(model)._denoise_flowmap_block.__globals__["trim_kv_cache"]
        self.N = model.bd_size

    def tau(self, u):
        """Pooled learned warp at base times u (any shape), endpoints pinned."""
        u = u.clamp(0, 1)
        t = self.w._invert_u_row(u.flatten().float().to(self.edges[0].device), *self.edges).view(u.shape)
        return torch.where(u <= 0, torch.zeros_like(t), torch.where(u >= 1, torch.ones_like(t), t))

    def grid(self, K, kind, offset=None):
        dev = self.m.embed_tokens.weight.device
        if kind is None:
            return self.m._get_generation_time_grid(K, dev, grid_mode="warp").float()  # [N, K+1]
        u = torch.linspace(0, 1, K + 1, device=dev)[None].repeat(self.N, 1)
        off = offset if offset is not None else self.c * (0.5 - torch.arange(self.N, device=dev) / (self.N - 1))
        if kind == "rev":
            off = -off
        u[:, 1:K] = u[:, 1:K] + off[:, None]
        return torch.cummax(self.tau(u).to(dev), dim=1).values

    @torch.no_grad()
    def draft(self, past, base_pos, K, kind, seed, latent=False):
        """Mirror of the checkpoint's `_denoise_flowmap_block` with a per-position grid, recording every jump."""
        m, N = self.m, self.N
        dtype, dev = m.embed_tokens.weight.dtype, m.embed_tokens.weight.device
        E = m.embed_tokens.weight
        grid = self.grid(K, None if kind == "conf" else kind)
        cache_len = past.get_seq_length()
        pos = torch.arange(base_pos, base_pos + N, device=dev)[None]
        mask = torch.zeros(1, 1, N, cache_len + N, device=dev, dtype=dtype)
        torch.manual_seed(seed)
        state = torch.randn(1, N, m._d_model, device=dev, dtype=dtype) * m._noise_sigma * m.noise_scale_mult
        rec = dict(argmax=[], p1=[], latent=[])
        for k in range(K):
            path = grid.to(dtype).T.unsqueeze(1)  # [K+1, 1, N]
            source, target = path[k], path[k + 1]
            cond = m._compose_flowmap_time_condition(source, target)
            logits, _, _ = m._forward_backbone(state, cond, mask, past_key_values=past, use_cache=False,
                                               position_ids=pos)
            self.trim(past, past.get_seq_length() - cache_len)
            p = F.softmax(logits[0].float(), dim=-1)
            p1, am = p.max(-1)
            rec["argmax"].append(am.tolist())
            rec["p1"].append([round(v, 4) for v in p1.tolist()])
            if k == K - 1:
                return rec, logits
            if kind == "conf" and k == 0:  # most confident slot ahead, from column 2 on
                rank = torch.argsort(torch.argsort(-p1)).float()
                g2 = self.grid(K, "l2r", self.c * (0.5 - rank / (N - 1)))
                grid = torch.cat([grid[:, :2], g2[:, 2:]], dim=1)
                grid = torch.cummax(grid, dim=1).values
            delta = p.to(dtype) @ E
            s, t = source.float().unsqueeze(-1), target.float().unsqueeze(-1)
            updated = (state.float() + (t - s) * (delta.float() - state.float()) / (1 - s).clamp_min(1e-6)).to(dtype)
            state = torch.where((target > source + 1e-8).unsqueeze(-1), updated, state)
            if latent:
                nn = (F.normalize(state[0].float(), dim=-1) @ F.normalize(E.float(), dim=-1).T).argmax(-1)
                rec["latent"].append(nn.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lafa-hf/lafa-7B")
    ap.add_argument("--task", choices=list(TASKS), default="gsm8k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=0.5, help="c: base-time offset span between positions 0 and 3")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--device", default="cuda", help="cpu only for a smoke test with lafa-0.5B")
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/lafa_e1_headroom")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
    model.noise_scale_mult = 0.1  # the HF `generate` default (noise_scale_override)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"model {args.model} bd={model.bd_size} two_time={model._two_time} dev={args.device}",
          flush=True)

    repo, cfg, split, field, instr = TASKS[args.task]
    rows = load_dataset(repo, cfg, split=split).select(range(args.n))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{args.task}.shard{args.shard}.jsonl")
    done = {json.loads(l)["idx"] for l in open(out)} if os.path.exists(out) else set()
    dr = Drafter(model, args.stagger)
    variants = args.variants.split(",")
    model.backbone.config._attn_implementation = "eager"  # dense block masks, as the checkpoint's own decoders
    checked = False

    with open(out, "a") as fh:
        for idx in range(args.shard, args.n, args.num_shards):
            if idx in done:
                continue
            row = rows[idx]
            prompt = tok.apply_chat_template([{"role": "user", "content": instr + row[field]}], tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
            sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
            sync()
            t0 = time.perf_counter()
            res = model.generate(input_ids=ids, gen_mode="speculative", max_new_tokens=args.max_new_tokens,
                                 grid_mode="warp", eos_token_id=eos, return_dict_in_generate=True)
            sync()
            spec_s = time.perf_counter() - t0
            model.backbone.config._attn_implementation = "eager"
            y = res.sequences[0, ids.shape[1]:].tolist()
            if eos in y:
                y = y[: y.index(eos) + 1]

            t0 = time.perf_counter()
            P = ids.shape[1]
            with torch.no_grad():
                ones = lambda n: model._compute_t_bn(torch.ones(1, n, device=ids.device, dtype=torch.bfloat16))
                logits, past, _ = model._forward_backbone(model.embed_tokens(ids), ones(P), attn_mask=None,
                                                          use_cache=True)
                ar_logit = (logits if logits.dim() == 3 else logits[None])[0, -1]
                ar_p1, ar_ent, ar_match, sites = [], [], [], []
                for p in range(len(y)):
                    lp = F.log_softmax(ar_logit.float(), -1)
                    ar_p1.append(round(lp.max().exp().item(), 4))
                    ar_ent.append(round(-(lp.exp() * lp).sum().item(), 4))
                    ar_match.append(int(lp.argmax().item() == y[p]))
                    if p % args.stride == 0:
                        site = dict(p=p)
                        for v in variants:
                            K, kind = VARIANTS[v]
                            rec, lg = dr.draft(past, P + p, K, kind, seed=idx * 100003 + p, latent=(v == "warp4"))
                            site[v] = rec
                            if not checked and v == "warp4":  # self-check against the checkpoint's own loop
                                torch.manual_seed(idx * 100003 + p)
                                ref, _ = model._denoise_flowmap_block(
                                    past, torch.arange(P + p, P + p + dr.N, device=ids.device)[None],
                                    model._get_generation_time_grid(4, ids.device, grid_mode="warp"),
                                    torch.zeros(1, 1, dr.N, past.get_seq_length() + dr.N, device=ids.device,
                                                dtype=torch.bfloat16))
                                same = torch.equal(ref[0].argmax(-1), lg[0].float().argmax(-1))
                                diff = (ref[0].float() - lg[0].float()).abs().max().item()
                                print(f"self-check warp4 vs _denoise_flowmap_block: argmax equal={same} "
                                      f"max|dlogit|={diff:.3g}", flush=True)
                                if not same:
                                    raise SystemExit("draft loop does not reproduce the checkpoint's flow-map block")
                                checked = True
                        sites.append(site)
                    emb = model.embed_tokens(torch.tensor([[y[p]]], device=ids.device))
                    lg, past, _ = model._forward_backbone(emb, ones(1), attn_mask=None, past_key_values=past,
                                                          use_cache=True,
                                                          position_ids=torch.tensor([[P + p]], device=ids.device))
                    ar_logit = (lg if lg.dim() == 3 else lg[None])[0, -1]
            r = dict(idx=idx, task=args.task, model=args.model, gold=str(row.get("answer", "")), prompt_len=P, y=y,
                     text=tok.decode(y, skip_special_tokens=True), spec_s=spec_s, probe_s=time.perf_counter() - t0,
                     spec_stats={k: v for k, v in res.stats[0].items() if isinstance(v, (int, float, bool))},
                     ar_p1=ar_p1, ar_ent=ar_ent, ar_match=ar_match, stride=args.stride, stagger=args.stagger,
                     sites=sites)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"{args.task} {idx}: len={len(y)} spec {spec_s:.1f}s probe {r['probe_s']:.1f}s "
                  f"ar_match={sum(ar_match)}/{len(y)} stats={r['spec_stats']}", flush=True)


if __name__ == "__main__":
    main()
