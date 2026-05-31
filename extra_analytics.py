"""Дополнительная аналитика поверх обученной LTR-модели.

Что строит:
- recall@k и nDCG@k кривые (LTR vs baseline_sim vs random) — обновлённый recall_at_n.png
- сравнение LTR vs baseline по entity_type (две группы баров рядом)
- uplift-таблица: на сколько LTR обгоняет baseline по основным метрикам
- partial dependence (sim, freshness, log_views) — как модель использует основные фичи
- calibration: распределение скора в clicked vs not_clicked (повторяет, но с двумя моделями)

Не переобучает модель — берёт сохранённую `models/ltr_lgbm.pkl`.

По умолчанию оценивает **только на i2i‑блоках** (`ml_what-else-mi-pisali` и
`what-else-mi-pisali`), чтобы убрать «грязь» от не‑i2i карусели
(`ml_personal`, `popularity-block`). Запуск с `--mixed` повторяет старую
оценку на смешанной test‑выборке.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from eval_ltr_i2i import I2I_ENTITY_TYPES
from train_ltr_full import (
    LTR_FEATURE_NAMES,
    FIGURES_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    build_dataset,
    fit_retriever,
    load_articles,
    load_coview_aligned,
    load_embeddings_memmap,
    load_logs,
    plot_feature_importance,
    plot_overall_metrics,
    plot_per_entity,
    time_split,
)


def predict_lgbm(model, X: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    return np.asarray(model.predict(df), dtype=np.float32)


def predict_catboost(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=np.float32)


def per_group_iter(y, group):
    pos = 0
    for gsz in group:
        sl = slice(pos, pos + gsz)
        pos += gsz
        yield gsz, y[sl]


def metrics_at_k(y, score, group, ks=(1, 3, 5, 6, 10), min_group_size: int = 2):
    out = {f"recall@{k}": [] for k in ks}
    out.update({f"ndcg@{k}": [] for k in ks})
    out.update({f"mrr@{k}": [] for k in ks})

    pos = 0
    for gsz in group:
        sl = slice(pos, pos + gsz)
        yi = y[sl]
        si = score[sl]
        pos += gsz
        if gsz < min_group_size or yi.sum() == 0:
            continue
        order = np.argsort(-si, kind="stable")
        ranked_y = yi[order]
        total_pos = float(yi.sum())
        for k in ks:
            top = ranked_y[:k]
            out[f"recall@{k}"].append(float(top.sum()) / total_pos)
            out[f"ndcg@{k}"].append(float(ndcg_score([yi], [si], k=k)))
            hits = np.where(top > 0)[0]
            out[f"mrr@{k}"].append(1.0 / (hits[0] + 1) if len(hits) else 0.0)
    return {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}, len(out["mrr@3"])


def plot_recall_curves(metrics_per_model: dict[str, dict[str, float]], ks: list[int], out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"lgbm": "#3478c6", "catboost": "#7c3acb", "baseline_sim": "#7d8c8c", "random": "#bbbbbb"}
    for name, vals in metrics_per_model.items():
        ys = [vals[f"recall@{k}"] for k in ks]
        ax.plot(ks, ys, marker="o", linewidth=2.0, label=name, color=palette.get(name, "#444"))
        for k, y in zip(ks, ys):
            ax.text(k, y + 0.012, f"{y:.2f}", ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("recall@K (внутри показа)")
    ax.set_title("Recall кликов из показа на test")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_ndcg_curves(metrics_per_model: dict[str, dict[str, float]], ks: list[int], out: Path):
    fig, ax = plt.subplots(figsize=(8.4, 5))
    palette = {"lgbm": "#3478c6", "catboost": "#7c3acb", "baseline_sim": "#7d8c8c", "random": "#bbbbbb"}
    all_y = []
    for name, vals in metrics_per_model.items():
        ys = [vals[f"ndcg@{k}"] for k in ks]
        all_y.extend(ys)
        ax.plot(ks, ys, marker="s", linewidth=2.0, label=name, color=palette.get(name, "#444"))
        for k, y in zip(ks, ys):
            ax.text(k, y + 0.012, f"{y:.3f}", ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("nDCG@K")
    ax.set_title("nDCG@K на test")
    lo = max(0.0, min(all_y) - 0.05)
    hi = max(all_y) + 0.05
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_per_entity_compare(merged: pd.DataFrame, k: int, out: Path):
    df = merged.sort_values("n_groups", ascending=False).head(8)
    et_short = [s.replace("article.", "") for s in df["entity_type"]]
    et_short = [(s[:34] + "…") if len(s) > 34 else s for s in et_short]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    width = 0.36
    ax.bar(x - width / 2, df[f"ndcg@{k}_baseline"], width=width, label="baseline (sim)", color="#7d8c8c")
    ax.bar(x + width / 2, df[f"ndcg@{k}"], width=width, label="LTR (LightGBM)", color="#3478c6")
    for i, (b, m, n) in enumerate(zip(df[f"ndcg@{k}_baseline"], df[f"ndcg@{k}"], df["n_groups"])):
        ax.text(i - width / 2, b + 0.008, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, m + 0.008, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i, max(b, m) + 0.04, f"n={int(n):,}\n+{(m - b)*100:.1f}pp", ha="center", fontsize=8.5, color="#1a1a1a")
    ax.set_xticks(x)
    ax.set_xticklabels(et_short, rotation=15, ha="right", fontsize=9)
    ax.set_title(f"nDCG@{k} по типу карусели: baseline vs LTR")
    ax.set_ylabel(f"nDCG@{k}")
    ax.set_ylim(0, max(df[f"ndcg@{k}"].max(), df[f"ndcg@{k}_baseline"].max()) * 1.18)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_uplift_summary(metrics: dict[str, dict[str, float]], out: Path):
    """Бар-чарт: на сколько процентов lgbm/catboost обгоняют baseline_sim
    по выбранным метрикам.
    """
    base = metrics["baseline_sim"]
    show = ["recall@3", "recall@6", "recall@10", "ndcg@3", "ndcg@6", "ndcg@10", "mrr@6"]
    models = [m for m in metrics if m not in ("baseline_sim", "random")]
    width = 0.4
    x = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {"lgbm": "#3478c6", "catboost": "#7c3acb"}
    for i, m in enumerate(models):
        ys = [(metrics[m][s] - base[s]) / max(base[s], 1e-6) * 100 for s in show]
        bars = ax.bar(x + (i - 0.5) * width, ys, width=width, label=m, color=palette.get(m, "#444"))
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + (0.5 if y >= 0 else -0.8),
                    f"{y:+.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(show, rotation=15)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("Прирост LTR vs baseline (semantic-only) на test, %")
    ax.set_ylabel("относительный прирост, %")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_score_distribution_single(score_pos, score_neg, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(min(score_neg.min(), score_pos.min()),
                       max(score_neg.max(), score_pos.max()), 60)
    ax.hist(score_neg, bins=bins, alpha=0.55, label="not clicked", color="#7d8c8c", density=True)
    ax.hist(score_pos, bins=bins, alpha=0.65, label="clicked", color="#f49a3b", density=True)
    ax.set_title("Распределение скоров LightGBM: clicked vs not clicked (test)")
    ax.set_xlabel("score")
    ax.set_ylabel("плотность")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_score_distribution_two(score_lgbm_pos, score_lgbm_neg, score_cb_pos, score_cb_neg, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, name, sp, sn in [
        (axes[0], "LightGBM", score_lgbm_pos, score_lgbm_neg),
        (axes[1], "CatBoost", score_cb_pos, score_cb_neg),
    ]:
        bins = np.linspace(min(sn.min(), sp.min()), max(sn.max(), sp.max()), 60)
        ax.hist(sn, bins=bins, alpha=0.55, label="not clicked", color="#7d8c8c", density=True)
        ax.hist(sp, bins=bins, alpha=0.65, label="clicked", color="#f49a3b", density=True)
        ax.set_title(name)
        ax.set_xlabel("score")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("плотность")
    fig.suptitle("Распределение скоров: clicked vs not clicked (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_partial_dependence(model, X: np.ndarray, feature: str, out: Path, *, n_points: int = 30):
    idx = LTR_FEATURE_NAMES.index(feature)
    base = X.copy()
    if X.shape[0] > 30_000:
        sel = np.random.default_rng(0).choice(X.shape[0], 30_000, replace=False)
        base = X[sel]
    grid = np.linspace(np.percentile(X[:, idx], 1), np.percentile(X[:, idx], 99), n_points)
    means = []
    df_template = pd.DataFrame(base.copy(), columns=LTR_FEATURE_NAMES)
    for v in grid:
        df_template[feature] = v
        means.append(float(np.mean(model.predict(df_template))))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, means, marker="o", color="#3478c6")
    ax.set_title(f"Partial dependence: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("средний LTR-скор")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def per_entity_table(y, score, group, et_per_group, k=6, min_group_size=2):
    by_et: dict[str, list[tuple[float, float]]] = {}
    pos = 0
    for gsz, et in zip(group, et_per_group.values):
        sl = slice(pos, pos + gsz)
        yi = y[sl]
        si = score[sl]
        pos += gsz
        if gsz < min_group_size or yi.sum() == 0:
            continue
        ndcg = float(ndcg_score([yi], [si], k=k))
        order = np.argsort(-si, kind="stable")
        ranked_y = yi[order][:k]
        hits = np.where(ranked_y > 0)[0]
        mrr = 1.0 / (hits[0] + 1) if len(hits) else 0.0
        by_et.setdefault(et, []).append((ndcg, mrr))
    rows = [{
        "entity_type": et,
        "n_groups": len(vals),
        f"ndcg@{k}": float(np.mean([v[0] for v in vals])),
        f"mrr@{k}": float(np.mean([v[1] for v in vals])),
    } for et, vals in by_et.items()]
    return pd.DataFrame(rows).sort_values("n_groups", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", action="store_true",
                    help="оценить на смешанной test‑выборке (включая ml_personal "
                         "и popularity-block). По умолчанию — только i2i.")
    args = ap.parse_args()
    i2i_only = not args.mixed
    mode_tag = "i2i" if i2i_only else "mixed"
    suffix = "_i2i" if i2i_only else ""
    print(f"[mode] evaluating on {'i2i blocks only' if i2i_only else 'mixed test (all entity types)'}")

    print("[load] articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)

    print("[load] LTR model + propensity…")
    bundle = pickle.loads((MODELS_DIR / "ltr_lgbm.pkl").read_bytes())
    model_lgbm = bundle["model"]

    print("[load] coview index…")
    _, _, coview_pair_df = load_coview_aligned(art)

    print("[load] logs…")
    logs = load_logs()
    _, test_logs, cut = time_split(logs, train_quantile=0.8)
    del logs

    if i2i_only:
        n_full = len(test_logs)
        test_logs = test_logs[test_logs["entity_type"].astype(str).isin(I2I_ENTITY_TYPES)].copy()
        n_i2i = len(test_logs)
        print(f"[filter] test rows: {n_full:,} → {n_i2i:,} (i2i only, {n_i2i/max(n_full,1)*100:.1f}%)")

    print("[build] test dataset…")
    Xte, yte, gte, _, _, et_te = build_dataset(
        art, test_logs, max_groups=150_000, coview_pair_df=coview_pair_df,
    )
    del test_logs

    s_lgbm = predict_lgbm(model_lgbm, Xte)
    s_baseline = Xte[:, 0]
    rng = np.random.default_rng(123)
    s_random = rng.standard_normal(len(yte)).astype(np.float32)

    s_cb = None
    cb_path = MODELS_DIR / "ltr_catboost.cbm"
    if cb_path.exists():
        try:
            from catboost import CatBoostRanker
            model_cb = CatBoostRanker()
            model_cb.load_model(str(cb_path))
            s_cb = predict_catboost(model_cb, Xte)
        except Exception as e:
            print("  (catboost skip:", e, ")")

    ks = [1, 3, 5, 6, 10]
    print("[metrics] computing…")
    res = {}
    res["lgbm"], n_eval = metrics_at_k(yte, s_lgbm, gte, ks=ks)
    res["baseline_sim"], _ = metrics_at_k(yte, s_baseline, gte, ks=ks)
    res["random"], _ = metrics_at_k(yte, s_random, gte, ks=ks)
    if s_cb is not None:
        res["catboost"], _ = metrics_at_k(yte, s_cb, gte, ks=ks)
    print(f"  groups evaluated: {n_eval:,}")

    (REPORTS_DIR / f"ltr_extra_metrics{suffix}.json").write_text(
        json.dumps({"groups_evaluated": n_eval, "mode": mode_tag, "metrics": res},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[plots]…")
    plot_recall_curves(res, ks, FIGURES_DIR / f"recall_at_n{suffix}.png")
    plot_ndcg_curves(res, ks, FIGURES_DIR / f"ndcg_at_n{suffix}.png")
    plot_uplift_summary(res, FIGURES_DIR / f"uplift_lgbm_vs_baseline{suffix}.png")

    pe_lgbm = per_entity_table(yte, s_lgbm, gte, et_te)
    pe_base = per_entity_table(yte, s_baseline, gte, et_te)
    merged = pe_lgbm.merge(
        pe_base.rename(columns={"ndcg@6": "ndcg@6_baseline", "mrr@6": "mrr@6_baseline"}).drop(columns=["n_groups"]),
        on="entity_type", how="left",
    )
    merged.to_csv(REPORTS_DIR / f"ltr_extra_per_entity{suffix}.csv", index=False)
    plot_per_entity_compare(merged, k=6, out=FIGURES_DIR / f"ltr_vs_baseline_by_entity{suffix}.png")

    if s_cb is not None:
        plot_score_distribution_two(
            s_lgbm[yte > 0], s_lgbm[yte == 0],
            s_cb[yte > 0], s_cb[yte == 0],
            FIGURES_DIR / f"score_distribution_two_models{suffix}.png",
        )

    # single-model score distribution (раньше делался в train_ltr_full.py на смешанной)
    plot_score_distribution_single(
        s_lgbm[yte > 0], s_lgbm[yte == 0],
        FIGURES_DIR / f"score_distribution_clicked_vs_not{suffix}.png",
    )

    print("[plots] partial dependence…")
    plot_partial_dependence(model_lgbm, Xte, "sim", FIGURES_DIR / f"pdp_sim{suffix}.png")
    plot_partial_dependence(model_lgbm, Xte, "cand_fresh", FIGURES_DIR / f"pdp_fresh{suffix}.png")
    plot_partial_dependence(model_lgbm, Xte, "cand_log_views", FIGURES_DIR / f"pdp_logviews{suffix}.png")

    # графики, которые раньше делались внутри train_ltr_full.py на смешанной выборке
    print("[plots] overall + per-entity + feature importance…")
    overall_order = [k for k in ("lgbm", "catboost", "baseline_sim", "random") if k in res]
    overall_metrics = {
        name: {f"{m}@{k}": res[name][f"{m}@{k}"] for m in ("ndcg", "mrr") for k in (3, 6, 10)}
        for name in overall_order
    }
    plot_overall_metrics(overall_metrics, FIGURES_DIR / f"ltr_metrics_overall{suffix}.png")
    plot_per_entity(pe_lgbm, FIGURES_DIR / f"ltr_metrics_by_entity{suffix}.png", k=6)
    plot_feature_importance(model_lgbm, LTR_FEATURE_NAMES,
                            FIGURES_DIR / f"feature_importance_gain{suffix}.png", kind="gain")

    print("\n=== DONE extra analytics ===")
    for m, vals in res.items():
        print(m, {k: round(v, 4) for k, v in vals.items()})


if __name__ == "__main__":
    main()
