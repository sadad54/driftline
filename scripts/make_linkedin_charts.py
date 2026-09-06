"""Generate the two data-driven LinkedIn post images directly from committed results/*.json --
genuinely reproducible charts, not mockups. Run: python scripts/make_linkedin_charts.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
OUT_DIR.mkdir(exist_ok=True)

NAVY = "#1a2332"
RED = "#e0574c"
GREEN = "#2ea87f"
GRAY = "#8a94a6"
BG = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": GRAY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})


def chart_decay_recovery():
    with open(RESULTS_DIR / "drift_triggered_retrain.json") as f:
        data = json.load(f)
    weekly = data["weekly_curve"]
    weeks = sorted(int(k.split("_")[1]) for k in weekly)
    values = [weekly[f"week_{w}"]["pr_auc_with_retraining"] for w in weeks]
    retrain_weeks = [e["trigger_week"] for e in data["retrain_events"]]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    ax.plot(weeks, values, color=NAVY, linewidth=2.5, zorder=3)
    ax.scatter(weeks, values, color=NAVY, s=18, zorder=4)

    for rw in retrain_weeks:
        ax.axvline(rw, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)

    # annotate the headline recovery: week 14 -> week 15
    w14, w15 = weekly["week_14"]["pr_auc_with_retraining"], weekly["week_15"]["pr_auc_with_retraining"]
    ax.annotate(
        f"retrain fires\nweek 14: {w14:.3f}",
        xy=(14, w14), xytext=(11.5, w14 - 0.08),
        ha="center", fontsize=10, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )
    ax.annotate(
        f"week 15: {w15:.3f}\n(+97.8% in one week)",
        xy=(15, w15), xytext=(18.5, w15 + 0.06),
        ha="left", fontsize=10, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
    )

    ax.set_title("Fraud model performance over a 6-month replay,\nwith drift-triggered retraining",
                  fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("PR-AUC", fontsize=11)
    ax.set_ylim(0.20, 0.65)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.7)
    fig.text(0.99, 0.01, "driftline -- real weekly PR-AUC, results/drift_triggered_retrain.json",
             ha="right", va="bottom", fontsize=7.5, color=GRAY, style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = OUT_DIR / "linkedin_1_decay_recovery.png"
    fig.savefig(out, facecolor=BG)
    print(f"Wrote {out}")


def chart_split_comparison():
    with open(RESULTS_DIR / "random_vs_time_split.json") as f:
        data = json.load(f)

    labels = ["Time-ordered split\n(honest)", "Random split\n(the anti-pattern)"]
    values = [data["time_ordered_split"]["pr_auc"], data["random_split"]["pr_auc"]]
    colors = [GREEN, RED]

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=200)
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                ha="center", fontsize=17, fontweight="bold", color=NAVY)

    inflation = data["pr_auc_inflation"]
    ax.annotate(
        f"+{inflation:.3f} inflation\nfrom the split alone",
        xy=(1, values[1] - 0.05), xytext=(0.5, 0.30),
        ha="center", fontsize=11, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    ax.set_title("Same model. Same data. Only the split changed.",
                  fontsize=14, fontweight="bold", pad=14, loc="left")
    ax.set_ylabel("PR-AUC", fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRAY, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.99, 0.01, "driftline -- results/random_vs_time_split.json",
             ha="right", va="bottom", fontsize=7.5, color=GRAY, style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = OUT_DIR / "linkedin_2_split_comparison.png"
    fig.savefig(out, facecolor=BG)
    print(f"Wrote {out}")


if __name__ == "__main__":
    chart_decay_recovery()
    chart_split_comparison()
