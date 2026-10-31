"""Backup-line follow-up, pilot P1: does branching at AR-uncertain tokens raise Lafa's accuracy (upper bound), and is
there a selection signal that captures it? (docs/PROGRESS.md, "Literature check: improving the AR part ...")

Lafa-7B, MATH-500. Greedy reference: self-speculative decoding (= AR greedy). Its AR top-1 probability per position
comes from one clean forward over prompt + output. A branch forces the AR second-ranked token at position p and lets
self-speculative decoding finish (same 512-token budget). Branch points are restricted to positions before the first
`\\boxed` (no direct answer guessing). Three policies, 4 branches each:
  low    the 4 positions with the lowest AR top-1
  early  the first 4 positions with AR top-1 < 0.5 (filled with the next-lowest if fewer)
  rand   4 random positions with AR top-1 >= 0.9 (control: any perturbation diversifies)
Per rollout: answer, correctness, forwards (decoder `total_steps`), and the mean AR log-probability of the whole
response (one clean forward) as a candidate selection signal. Read-outs (diagnostics): any-correct per policy vs
greedy (upper bound), majority vote and max-log-probability selection vs greedy, forwards per problem.

    python src/scripts/lafa_branch_headroom.py --n 200 --shard 0 --num-shards 2
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.scripts.lafa_draft_headroom import TASKS  # noqa: E402
from src.scripts.lafa_headroom_report import boxed, correct  # noqa: E402

ACCT = os.popen("id -un").read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lafa-hf/lafa-7B")
    ap.add_argument("--task", default="math-500")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda", help="cpu only for a smoke test with lafa-0.5B")
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/lafa_p1_branch")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    repo, cfg, split, field, instr = TASKS[args.task]
    rows = load_dataset(repo, cfg, split=split).select(range(args.n))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"{args.task}.shard{args.shard}.jsonl")
    done = {json.loads(l)["idx"] for l in open(out)} if os.path.exists(out) else set()

    @torch.no_grad()
    def ar_logits(ids):
        lg = model(input_ids=ids).logits
        return (lg if lg.dim() == 3 else lg[None])[0].float()  # [L, V], position j predicts j+1

    def rollout(prompt, resp_prefix, budget):
        ids = torch.tensor([prompt + resp_prefix], device=args.device)
        sync()
        t0 = time.perf_counter()
        res = model.generate(input_ids=ids, gen_mode="speculative", max_new_tokens=max(budget, 1),
                             grid_mode="warp", eos_token_id=eos, return_dict_in_generate=True)
        sync()
        cont = res.sequences[0, ids.shape[1]:].tolist()
        if eos in cont:
            cont = cont[: cont.index(eos) + 1]
        resp = resp_prefix + cont
        lp = F.log_softmax(ar_logits(torch.tensor([prompt + resp], device=args.device)), -1)
        P = len(prompt)
        tgt = torch.tensor(resp, device=args.device)
        mean_lp = lp[P - 1:P - 1 + len(resp)].gather(1, tgt[:, None]).mean().item()
        text = tok.decode(resp, skip_special_tokens=True)
        return dict(resp_len=len(resp), forwards=int(res.stats[0]["total_steps"]), s=time.perf_counter() - t0,
                    mean_lp=round(mean_lp, 5), answer=boxed(text), text=text), resp

    with open(out, "a") as fh:
        for idx in range(args.shard, args.n, args.num_shards):
            if idx in done:
                continue
            row = rows[idx]
            gold = str(row["answer"])
            prompt = tok(tok.apply_chat_template([{"role": "user", "content": instr + row[field]}], tokenize=False,
                                                 add_generation_prompt=True)).input_ids
            g, y = rollout(prompt, [], args.max_new_tokens)
            g["correct"] = correct(args.task, g["text"], gold)
            P = len(prompt)
            lp = F.log_softmax(ar_logits(torch.tensor([prompt + y], device=args.device)), -1)[P - 1:P - 1 + len(y)]
            top = lp.topk(2, -1)
            p1 = top.values[:, 0].exp().tolist()
            alt = top.indices[:, 1].tolist()
            # last position allowed: before the first \boxed (binary search on the decoded prefix)
            lo, hi = 0, len(y)
            while lo < hi:
                mid = (lo + hi) // 2
                if "\\boxed" in tok.decode(y[:mid + 1]):
                    hi = mid
                else:
                    lo = mid + 1
            limit = lo if lo < len(y) else max(len(y) - 10, 1)
            cand = list(range(limit))
            low = sorted(cand, key=lambda p: p1[p])[:args.branches]
            early = [p for p in cand if p1[p] < 0.5][:args.branches]
            early += [p for p in sorted(cand, key=lambda p: p1[p]) if p not in early][:args.branches - len(early)]
            conf = [p for p in cand if p1[p] >= 0.9]
            rand = sorted(random.Random(idx).sample(conf, min(args.branches, len(conf))))
            policies = dict(low=sorted(low), early=sorted(early), rand=rand)
            cache, branches = {}, []
            for pol, ps in policies.items():
                for p in ps:
                    if p not in cache:
                        r, _ = rollout(prompt, y[:p] + [alt[p]], args.max_new_tokens - p - 1)
                        r.update(p=p, p1=round(p1[p], 4), alt=alt[p], correct=correct(args.task, r["text"], gold))
                        cache[p] = r
                    branches.append(dict(policy=pol, **cache[p]))
            rec = dict(idx=idx, task=args.task, model=args.model, gold=gold, prompt_len=P, greedy=g,
                       boxed_limit=limit, n_p1_below_05=sum(q < 0.5 for q in p1[:limit]), branches=branches)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ac = {pol: any(b["correct"] for b in branches if b["policy"] == pol) for pol in policies}
            print(f"{idx}: greedy={g['correct']} len={len(y)} limit={limit} any={ac} "
                  f"rollouts={len(cache)} s={sum(c['s'] for c in cache.values()) + g['s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
