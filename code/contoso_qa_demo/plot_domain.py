"""Plot the domain-LoRA contrast (base vs base+prompt vs adapter), grayscale.

Reads runs/domain_lora/results.json and writes runs/domain_lora/contrast.png/svg.
Headline panel: internal-terminology use, where the adapter should tower over
base and base+prompt (the gain a prompt cannot supply).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DARK, MID, LIGHT = "#4d4d4d", "#7a7a7a", "#bdbdbd"
MODELS = ["base", "base+prompt", "adapter"]
METRICS = [("internal_tool_in_body", "Internal tool named\n(in answer body)"),
           ("house_format", "House format\n(steps + escalation)"),
           ("token_f1", "Token-F1\nvs reference")]


def main():
    res = json.loads((HERE / "runs/domain_lora/results.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = range(len(METRICS))
    w = 0.26
    shades = {"base": LIGHT, "base+prompt": MID, "adapter": DARK}
    for i, m in enumerate(MODELS):
        vals = [res[m][k] for k, _ in METRICS]
        bars = ax.bar([xi + (i - 1) * w for xi in x], vals, w, label=m,
                      color=shades[m], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in METRICS], fontsize=7)
    ax.set_ylabel("score (0 to 1)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = HERE / "runs/domain_lora/contrast.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
