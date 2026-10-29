"""Per-image analysis of a fixed-interval frontier campaign (bad cases, divergence point, cross-arm agreement).

    python src/scripts/analyze_frontier.py [--gens GENERATIONS] [--manifest MANIFEST] [--out SUMMARY_JSON]
                                           [--figs FIG_DIR]

Reads the campaign's per-sample generations (one row per arm x sample) and writes a summary json plus the figures
of docs/experiments/exp3_* and obs1_*. Jaccard is gate_check's word-set Jaccard. CPU only; runs on the
interactive node in the llada-v env.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from gate_check import ACCT, OUT, PAPER, PAPER_FULL_S, jaccard  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa", "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
OURS, PAPER_C, ACCENT = "#2a6fdb", "#9a9a9a", "#d1495b"
KS = [1, 2, 4, 8, 16, 32]


def boot_ci(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(v), 3) for v in np.percentile(means, [2.5, 97.5])]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", default=f"{OUT}/mme_caption_frontier/generations.jsonl")
    ap.add_argument("--manifest", default=f"/scratch/{ACCT}/data/dmllm_refresh/mme_caption/manifest.jsonl")
    ap.add_argument("--out", default="results/phase1/mme_caption_frontier/analysis_summary.json")
    ap.add_argument("--figs", default="docs/experiments/figures")
    args = ap.parse_args()

    by = defaultdict(dict)
    for r in map(json.loads, open(args.gens)):
        by[r["arm"]][r["sample_id"]] = r
    meta = {r["sample_id"]: r for r in map(json.loads, open(args.manifest))}
    ids = sorted(by["full"])
    arms = ["full"] + [f"interval{k}" for k in KS]
    full = by["full"]

    jac = {k: np.array([jaccard(by[f"interval{k}"][i]["text"], full[i]["text"]) for i in ids]) for k in KS}
    pre = {k: np.array([common_prefix(by[f"interval{k}"][i]["tokens"], full[i]["tokens"]) for i in ids]) for k in KS}
    exact = {k: np.array([by[f"interval{k}"][i]["tokens"] == full[i]["tokens"] for i in ids]) for k in KS}
    n_prompt = np.array([full[i]["n_prompt"] for i in ids])
    lengths = {a: [len(by[a][i]["tokens"]) for i in ids] for a in arms}
    cross = [[float(np.mean([jaccard(by[a][i]["text"], by[b][i]["text"]) for i in ids])) for b in arms] for a in arms]
    rank_corr = spearmanr(np.stack([jac[k] for k in KS]).T).correlation

    cats = defaultdict(list)
    for j, i in zip(jac[2], ids):
        cats[meta[i]["category"]].append(j)

    i1_off = ~exact[1]
    summary = dict(
        n=len(ids), source=args.gens,
        note="Diagnostic analysis of the G1(ii) frontier; per-image numbers steer decisions, the headline stays g1_summary.json",
        output_lengths={a: sorted(set(v)) for a, v in lengths.items()},
        per_interval={k: dict(
            jaccard_mean=round(float(jac[k].mean()), 3), jaccard_ci95=boot_ci(jac[k]),
            jaccard_sd=round(float(jac[k].std(ddof=1)), 3),
            jaccard_min=round(float(jac[k].min()), 3), exact_match=int(exact[k].sum()),
            prefix_median=float(np.median(pre[k])), prefix_q25_q75=[float(v) for v in np.percentile(pre[k], [25, 75])],
            frac_prefix_le_16=round(float((pre[k] <= 16).mean()), 2),
            spearman_jaccard_vs_prefix=round(float(spearmanr(jac[k], pre[k]).correlation), 2),
            spearman_jaccard_vs_n_prompt=round(float(spearmanr(jac[k], n_prompt).correlation), 2),
            forwards_mean=round(float(np.mean([by[f"interval{k}"][i]["forwards"] for i in ids])), 1),
            refreshes_mean=round(float(np.mean([by[f"interval{k}"][i]["refreshes"] for i in ids])), 1),
        ) for k in KS},
        interval1_nonidentical=dict(n=int(i1_off.sum()), jaccard=[round(float(v), 3) for v in jac[1][i1_off]],
                                    prefix=[int(v) for v in pre[1][i1_off]]),
        cross_arm_jaccard=dict(arms=arms, matrix=[[round(v, 3) for v in row] for row in cross]),
        spearman_per_image_jaccard_across_intervals=dict(
            intervals=KS, matrix=[[round(float(v), 2) for v in row] for row in rank_corr]),
        interval2_jaccard_by_category={c: dict(n=len(v), mean=round(float(np.mean(v)), 3))
                                       for c, v in sorted(cats.items(), key=lambda kv: np.mean(kv[1]))},
        interval2_worst=[dict(sample_id=ids[j], mme_id=meta[ids[j]]["mme_id"], jaccard=round(float(jac[2][j]), 3),
                              prefix=int(pre[2][j]), n_prompt=int(n_prompt[j])) for j in np.argsort(jac[2])[:8]],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    os.makedirs(args.figs, exist_ok=True)
    full_s = sum(full[i]["gen_s"] for i in ids)
    speed = [full_s / sum(by[f"interval{k}"][i]["gen_s"] for i in ids) for k in KS]
    cis = np.array([summary["per_interval"][k]["jaccard_ci95"] for k in KS])
    means = np.array([jac[k].mean() for k in KS])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([PAPER_FULL_S / PAPER[k][0] for k in KS], [PAPER[k][1] for k in KS], "s--", color=PAPER_C,
            label="Faster but Different Tab. 3 (RTX 6000 Ada)")
    ax.errorbar(speed, means, yerr=[means - cis[:, 0], cis[:, 1] - means], fmt="o-", color=OURS, capsize=3,
                label="ours, 50 MME images (A100 80GB), 95% CI")
    for k, x, y in zip(KS, speed, means):
        ax.annotate(f"k={k}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("speedup over full recompute (wall-clock)")
    ax.set_ylabel("word-set Jaccard vs full output")
    ax.legend(fontsize=9)
    fig.savefig(f"{args.figs}/exp3_frontier_speed_jaccard.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for x, k in enumerate(KS):
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(ids)), jac[k], s=12, alpha=0.6, color=OURS)
        ax.plot([x - 0.3, x + 0.3], [jac[k].mean()] * 2, color=ACCENT, lw=2)
    ax.set_xticks(range(len(KS)), [f"k={k}" for k in KS])
    ax.set_ylabel("per-image Jaccard vs full")
    ax.set_xlabel("refresh interval (red bar = mean)")
    fig.savefig(f"{args.figs}/obs1_perimage_jaccard.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for k, c in [(2, OURS), (8, ACCENT)]:
        ax.scatter(pre[k], jac[k], s=16, alpha=0.7, color=c,
                   label=f"k={k} (Spearman {summary['per_interval'][k]['spearman_jaccard_vs_prefix']})")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("first divergent answer position (tokens shared with full)")
    ax.set_ylabel("Jaccard vs full")
    ax.legend(fontsize=9)
    fig.savefig(f"{args.figs}/obs1_prefix_vs_jaccard.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    m = np.array(cross)
    im = ax.imshow(m, cmap="Blues", vmin=0.3, vmax=1.0)
    labels = ["full"] + [f"k={k}" for k in KS]
    ax.set_xticks(range(len(arms)), labels, rotation=45)
    ax.set_yticks(range(len(arms)), labels)
    ax.grid(False)
    for a in range(len(arms)):
        for b in range(len(arms)):
            ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[a, b] > 0.75 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean word-set Jaccard")
    fig.savefig(f"{args.figs}/obs1_cross_arm_jaccard.png")
    plt.close(fig)
    print(json.dumps(summary["per_interval"], indent=1))


if __name__ == "__main__":
    main()
