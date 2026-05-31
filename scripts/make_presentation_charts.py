"""Чистые графики LTR vs baseline для предзащиты/презентации.

По умолчанию берёт i2i‑метрики из `reports/ltr_extra_metrics_i2i.json`
(получены через `eval_ltr_i2i.py` — оценка только на i2i‑блоках:
`ml_what-else-mi-pisali` и `what-else-mi-pisali`). Это методологически
правильная оценка для нашей i2i‑постановки. Чтобы построить графики на
старой смешанной выборке (включая не‑i2i блоки `ml_personal` и
`popularity-block`), запустите с `--mixed`.

Что строит:
  1) nDCG@K (K=1,3,5,6,10) — линии по моделям;
  2) MRR@K — то же;
  3) Recall@K — то же;
  4) сводная диаграмма приростов LightGBM/CatBoost над baseline (cosine);
  5) сводная таблица‑картинка по 4 моделям.

Выход: reports/figures/presentation/*.png, DPI=200, светлая тема.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OUT = REPORTS / "figures" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--mixed", action="store_true",
                help="использовать старую смешанную выборку (ltr_extra_metrics.json) "
                     "вместо чистой i2i‑выборки (ltr_extra_metrics_i2i.json)")
args = ap.parse_args()

if args.mixed:
    metrics_path = REPORTS / "ltr_extra_metrics.json"
    eval_label = "смешанная выборка (включая non‑i2i блоки)"
else:
    metrics_path = REPORTS / "ltr_extra_metrics_i2i.json"
    eval_label = "только i2i‑блоки"

EXTRA = json.loads(metrics_path.read_text())
m = EXTRA["metrics"]
n_groups = EXTRA.get("groups_evaluated", 0)
print(f"[charts] using {metrics_path.name}  ({eval_label}, {n_groups:,} групп)")

KS = [1, 3, 5, 6, 10]
MODELS_ORDER = [
    ("lgbm", "LightGBM (LambdaRank)", "#1f6feb"),
    ("catboost", "CatBoost (YetiRank)", "#0a7a3c"),
    ("baseline_sim", "Baseline — cosine kNN", "#d04a02"),
    ("random", "Random", "#8a8a8a"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def line_metric(metric_key: str, ylabel: str, title: str, fname: str, ylim: tuple | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    for key, label, color in MODELS_ORDER:
        ys = [m[key][f"{metric_key}@{k}"] for k in KS]
        marker = "o" if key in ("lgbm", "catboost") else "s"
        lw = 2.8 if key in ("lgbm", "catboost") else 1.8
        ls = "-" if key in ("lgbm", "catboost") else "--"
        ax.plot(KS, ys, marker=marker, lw=lw, ls=ls, color=color, label=label, markersize=9)
        if key == "lgbm":
            for k_val, y in zip(KS, ys):
                if k_val == 1:
                    xt, ha = (8, 4), "left"
                elif k_val == 10:
                    xt, ha = (-8, 6), "right"
                else:
                    xt, ha = (0, 12), "center"
                ax.annotate(f"{y:.3f}", (k_val, y), textcoords="offset points",
                            xytext=xt, ha=ha, fontsize=11, color=color, weight="bold")
    ax.set_xticks(KS)
    ax.set_xlabel("K — глубина оценки")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


line_metric("ndcg", "nDCG@K", "nDCG@K: LTR обходит baseline (cosine sim)",
            "01_ndcg_vs_baseline.png", ylim=(0.30, 0.79))
line_metric("mrr", "MRR@K", "MRR@K: ранжирующая модель улучшает позицию первого клика",
            "02_mrr_vs_baseline.png", ylim=(0.30, 0.72))
line_metric("recall", "Recall@K",
            "Recall@K: LTR находит ~40% релевантных уже на 1‑й позиции",
            "03_recall_vs_baseline.png", ylim=(0.28, 1.02))


# --- 4. сводная диаграмма приростов (relative %) -----------------------------
KEY_METRICS = [
    ("recall@1", "Recall@1"),
    ("ndcg@3", "nDCG@3"),
    ("ndcg@6", "nDCG@6"),
    ("mrr@3", "MRR@3"),
    ("mrr@6", "MRR@6"),
]

base = m["baseline_sim"]
lgbm = m["lgbm"]
cat = m["catboost"]

uplift_lgbm = [(lgbm[k] - base[k]) / base[k] * 100 for k, _ in KEY_METRICS]
uplift_cat = [(cat[k] - base[k]) / base[k] * 100 for k, _ in KEY_METRICS]

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
x = np.arange(len(KEY_METRICS))
width = 0.36

bars1 = ax.bar(x - width / 2, uplift_lgbm, width, label="LightGBM", color="#1f6feb")
bars2 = ax.bar(x + width / 2, uplift_cat, width, label="CatBoost", color="#0a7a3c")

for bars, vals in [(bars1, uplift_lgbm), (bars2, uplift_cat)]:
    for b, v in zip(bars, vals):
        ax.annotate(f"+{v:.1f}%", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, weight="bold")

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in KEY_METRICS])
ax.set_ylabel("Прирост над baseline (cosine sim), %")
ax.set_title("Относительный прирост LTR над baseline")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(0, max(max(uplift_lgbm), max(uplift_cat)) * 1.18)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "04_uplift_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# --- 5. компактная сводная таблица-картинка (для слайда вместо большой таблицы)
fig, ax = plt.subplots(figsize=(13, 4.5), dpi=200)
ax.axis("off")
cols = ["Модель", "nDCG@3", "nDCG@6", "MRR@3", "MRR@6", "Recall@1", "Recall@6"]
rows = []
for key, label, color in MODELS_ORDER:
    rows.append([
        label,
        f"{m[key]['ndcg@3']:.3f}",
        f"{m[key]['ndcg@6']:.3f}",
        f"{m[key]['mrr@3']:.3f}",
        f"{m[key]['mrr@6']:.3f}",
        f"{m[key]['recall@1']:.3f}",
        f"{m[key]['recall@6']:.3f}",
    ])

# первая колонка шире, чтобы названия моделей не обрезались
col_widths = [0.34, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=col_widths)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 2.0)
for i in range(len(cols)):
    table[(0, i)].set_facecolor("#0b1d3a")
    table[(0, i)].get_text().set_color("white")
    table[(0, i)].get_text().set_weight("bold")
for r, (key, _, color) in enumerate(MODELS_ORDER, start=1):
    table[(r, 0)].get_text().set_color(color)
    table[(r, 0)].get_text().set_weight("bold")
    table[(r, 0)].set_text_props(ha="left")
    table[(r, 0)].PAD = 0.04
ax.set_title("Итоговые оффлайн‑метрики на тестовой выборке",
             fontsize=14, pad=18)
fig.tight_layout()
fig.savefig(OUT / "05_summary_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"saved into: {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
