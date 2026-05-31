"""Честная аналитика propensity по логам.

Что строит:
- propensity_by_position.png — обновлённый график общий (на train),
  с подписями и комментарием в caption;
- propensity_by_entity.png — propensity отдельно для каждого типа
  карусели (одна линия = один entity_type);
- propensity_by_entity.csv — табличка с числами для приложения.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_ltr_full import (
    FIGURES_DIR,
    REPORTS_DIR,
    load_logs,
    time_split,
)


def estimate(group: pd.DataFrame, alpha: float = 1.0, beta: float = 1.0) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    g = group.groupby("article_position", observed=True, sort=True)
    for pos, vals in g:
        n = len(vals)
        k = int(vals["target"].sum())
        out[int(pos)] = {
            "n": int(n),
            "clicks": int(k),
            "p_hat": (k + alpha) / (n + alpha + beta),
        }
    return out


def plot_overall(p: dict[int, dict[str, float]], out: Path) -> None:
    items = sorted(p.items())
    xs = [str(k) for k, _ in items]
    ys = [v["p_hat"] for _, v in items]
    ns = [v["n"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(xs, ys, color="#3478c6")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y, f"{y:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Доля кликов по позициям на train (усреднение по всем типам каруселей)")
    ax.set_xlabel("article_position")
    ax.set_ylabel("p̂(click | position)")
    ax.set_ylim(0, max(ys) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_by_entity(per_et: dict[str, dict[int, dict[str, float]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    palette = ["#3478c6", "#f49a3b", "#7c3acb", "#3aa852", "#d96323", "#7d8c8c"]
    for i, (et, p) in enumerate(per_et.items()):
        items = sorted(p.items())
        xs = [k for k, _ in items]
        ys = [v["p_hat"] for _, v in items]
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=et.replace("article.", ""),
                color=palette[i % len(palette)])
        for x, y in zip(xs, ys):
            ax.text(x, y, f" {y:.3f}", fontsize=8.5, color="#333", va="center")
    ax.set_title("p̂(click | position) в разрезе типа карусели")
    ax.set_xlabel("article_position")
    ax.set_ylabel("p̂(click | position)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    print("[load] logs…")
    logs = load_logs()
    train_logs, _, cut = time_split(logs, train_quantile=0.8)
    del logs

    print("[propensity] overall…")
    p_all = estimate(train_logs)

    plot_overall(p_all, FIGURES_DIR / "propensity_by_position.png")
    print(f"  saved {FIGURES_DIR/'propensity_by_position.png'}")

    print("[propensity] by entity_type…")
    per_et: dict[str, dict[int, dict[str, float]]] = {}
    rows = []
    for et, sub in train_logs.groupby("entity_type", observed=True):
        if len(sub) < 5_000:
            continue
        p = estimate(sub)
        per_et[str(et)] = p
        for pos, v in sorted(p.items()):
            rows.append({"entity_type": str(et), "position": pos,
                         "n": v["n"], "clicks": v["clicks"], "p_hat": v["p_hat"]})

    plot_by_entity(per_et, FIGURES_DIR / "propensity_by_entity.png")
    print(f"  saved {FIGURES_DIR/'propensity_by_entity.png'}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "propensity_by_entity.csv", index=False)
    print(f"  saved {REPORTS_DIR/'propensity_by_entity.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
