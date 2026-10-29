"""Summarize the Phase 2 replacement-error probe (src/scripts/probe_replacement.py): G2 verdict, D3 time profile,
D1 group comparison, layer bands, decoding order.

    python src/scripts/analyze_probe.py [--rows ROWS_JSONL] [--out SUMMARY_JSON] [--figs FIG_DIR]

G2 (cell definition fixed in docs/PROGRESS.md before the run): cells = {visual, text} x 4 layer bands x step,
error = top-1 flips summed over images at age 4; PASS if the top 20% of cells hold >= 50% of flips. Steps are
bucketed by relative position (step / steps of that image, 10 bins) so images with different step counts align.
Prints `GATE G2 PASS|FAIL`. Diagnostic numbers only, never reported as quality.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gate_check import OUT  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa", "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})
BANDS = ["L0-7", "L8-15", "L16-23", "L24-31"]
NBIN = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=f"{OUT}/phase2_probe_mme/rows.jsonl")
    ap.add_argument("--out", default="results/phase2/probe_mme/probe_summary.json")
    ap.add_argument("--figs", default="docs/experiments/figures")
    ap.add_argument("--tag", default="exp5")
    args = ap.parse_args()

    rows, ends = defaultdict(list), {}
    for r in map(json.loads, open(args.rows)):
        (ends.__setitem__(r["sample_id"], r) if r.get("end") else rows[r["sample_id"]].append(r))
    ids = sorted(ends)

    # flips[key][bin], masked[bin], decision changes[key][bin], steps[bin]
    flips, kl, dec = defaultdict(lambda: np.zeros(NBIN)), defaultdict(lambda: np.zeros(NBIN)), defaultdict(lambda: np.zeros(NBIN))
    seen = defaultdict(lambda: np.zeros(NBIN))
    masked = np.zeros(NBIN)
    g2_cells = defaultdict(float)
    decoded_bin = np.zeros(NBIN)
    for i in ids:
        steps = sorted(rows[i], key=lambda r: r["t"])
        for r in steps:
            b = min(NBIN - 1, r["t"] * NBIN // len(steps))
            masked[b] += r["n_masked"]
            decoded_bin[b] += len(r["decoded_prev"])
            for key, (f, k, d) in r["cells"].items():
                flips[key][b] += f
                kl[key][b] += k
                dec[key][b] += d
                seen[key][b] += 1
                g, band, s = key.split("|")
                if s == "4" and g != "all":
                    g2_cells[(g, band, i, r["t"])] += f

    # G2 over (group x band x step) cells, summed over images per step index
    per_cell = defaultdict(float)
    for (g, band, i, t), f in g2_cells.items():
        per_cell[(g, band, t)] += f
    v = np.sort(np.array(list(per_cell.values())))[::-1]
    top = max(1, int(round(0.2 * len(v))))
    share = float(v[:top].sum() / v.sum()) if v.sum() else 0.0

    def frac_first(key, nb=2):
        return float(flips[key][:nb].sum() / flips[key].sum()) if flips[key].sum() else 0.0

    n_vis = np.mean([ends[i]["n_visual"] for i in ids])
    n_txt = np.mean([ends[i]["n_prompt"] - ends[i]["n_visual"] for i in ids])
    summary = dict(
        n=len(ids), rows=args.rows,
        mean_n_prompt=round(float(np.mean([ends[i]["n_prompt"] for i in ids])), 1),
        mean_n_visual=round(float(n_vis), 1), mean_probe_s=round(float(np.mean([ends[i]["probe_s"] for i in ids])), 1),
        g2=dict(cells=len(v), top20_share=round(share, 3), pass_=share >= 0.5, age=4),
        d3_all_layers={s: dict(flip_rate_by_bin=[round(float(x), 4) for x in flips[f"all|all|{s}"] / masked],
                               decision_change_rate_by_bin=[round(float(x), 3) for x in dec[f"all|all|{s}"] / np.maximum(seen[f"all|all|{s}"], 1)],
                               decision_change_rate=round(float(dec[f"all|all|{s}"].sum() / max(seen[f"all|all|{s}"].sum(), 1)), 3),
                               first_20pct_share_of_flips=round(frac_first(f"all|all|{s}"), 3))
                       for s in (1, 2, 4, 8)},
        d1_groups={s: {g: dict(flips=int(sum(flips[f"{g}|{b}|{s}"].sum() for b in BANDS)),
                               decision_changes=int(sum(dec[f"{g}|{b}|{s}"].sum() for b in BANDS)))
                       for g in ("visual", "text")} for s in (1, 4)},
        bands={s: {b: int(flips[f"visual|{b}|{s}"].sum() + flips[f"text|{b}|{s}"].sum()) for b in BANDS} for s in (1, 4)},
        tokens_decoded_by_bin=[int(x) for x in decoded_bin],
    )
    for s in (1, 4):
        d = summary["d1_groups"][s]
        d["flips_per_position_ratio_text_over_visual"] = round(
            (d["text"]["flips"] / n_txt) / max(d["visual"]["flips"] / n_vis, 1e-9), 2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    os.makedirs(args.figs, exist_ok=True)
    x = (np.arange(NBIN) + 0.5) / NBIN
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for s, c in zip((1, 2, 4, 8), ("#9ecae1", "#6baed6", "#2171b5", "#08306b")):
        axes[0].plot(x, flips[f"all|all|{s}"] / masked, "o-", color=c, label=f"age {s}")
        axes[1].plot(x, dec[f"all|all|{s}"] / np.maximum(seen[f"all|all|{s}"], 1), "o-", color=c, label=f"age {s}")
    axes[0].set_ylabel("top-1 flip rate (masked positions)")
    axes[1].set_ylabel("share of steps whose decision changes")
    for a in axes:
        a.set_xlabel("relative denoising step")
        a.legend(fontsize=9)
    fig.savefig(f"{args.figs}/{args.tag}_d3_time_profile.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    for j, (g, c) in enumerate((("visual", "#2a6fdb"), ("text", "#d1495b"))):
        ax.bar(np.arange(4) + (j - 0.5) * w, [flips[f"{g}|{b}|4"].sum() for b in BANDS], w, color=c, label=g)
    ax.set_xticks(range(4), BANDS)
    ax.set_ylabel("top-1 flips at age 4 (all images)")
    ax.legend()
    fig.savefig(f"{args.figs}/{args.tag}_group_band_flips.png")
    plt.close(fig)

    print(json.dumps(summary, indent=1))
    print(f"GATE G2 {'PASS' if share >= 0.5 else 'FAIL'} (captions, age 4): top 20% of {len(v)} cells hold {share:.1%} of flips")


if __name__ == "__main__":
    main()
