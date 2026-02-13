#!/usr/bin/env python3
"""Generate a log-log throughput plot from benchmark results."""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "throughput_results.csv")
PLOT_PATH = os.path.join(SCRIPT_DIR, "..", "docs", "throughput.png")

# Colors and markers per algorithm
ALGO_STYLE = {
    "CE":  {"color": "#1f77b4", "marker": "o"},
    "AOV": {"color": "#ff7f0e", "marker": "s"},
    "LS":  {"color": "#2ca02c", "marker": "^"},
    "FPW": {"color": "#d62728", "marker": "D"},
    "BLS": {"color": "#9467bd", "marker": "v"},
}

def load_results(csv_path):
    """Load CSV into nested dict: data[backend][algo] = {n_points: [...], throughput: [...]}"""
    data = defaultdict(lambda: defaultdict(lambda: {"n_points": [], "throughput": []}))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row["backend"]
            algo = row["algorithm"]
            data[backend][algo]["n_points"].append(int(row["n_points"]))
            data[backend][algo]["throughput"].append(float(row["throughput_pts_per_sec"]))
    return data


def main():
    data = load_results(CSV_PATH)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for backend in ["GPU", "CPU"]:
        if backend not in data:
            continue
        linestyle = "-" if backend == "GPU" else "--"
        for algo in ["CE", "AOV", "LS", "FPW", "BLS"]:
            if algo not in data[backend]:
                continue
            d = data[backend][algo]
            style = ALGO_STYLE[algo]
            label = f"{algo} ({backend})"
            ax.plot(
                d["n_points"], d["throughput"],
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=label,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Points per light curve", fontsize=13)
    ax.set_ylabel("Throughput (points / sec)", fontsize=13)
    ax.set_title(
        f"Periodfind throughput — {data['CPU']['CE']['n_points'][0]}–"
        f"{data['CPU']['CE']['n_points'][-1]} pts/curve, "
        f"100 curves × 1000 periods",
        fontsize=14,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper left")

    # Custom x-tick labels
    xticks = sorted(set(
        n for b in data.values() for a in b.values() for n in a["n_points"]
    ))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
