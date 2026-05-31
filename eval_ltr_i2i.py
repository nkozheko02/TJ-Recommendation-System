"""Чистая оценка LTR только на i2i‑блоках.

Берёт уже обученные модели (`models/ltr_lgbm.pkl`, `models/ltr_catboost.cbm`)
и пересчитывает все метрики на тестовых impression'ах, отфильтрованных по
i2i entity_type. Это убирает «грязь» от не‑i2i блоков (`ml_personal`,
`popularity-block`), на которых ранжирование по cosine sim к источнику не
имеет смысла и баseline вырождается до random.

Сохраняет:
  - reports/ltr_full_metrics_i2i.json     — сводные nDCG/MRR overall + per_entity
  - reports/ltr_extra_metrics_i2i.json    — recall/ndcg/mrr @1,3,5,6,10 для всех моделей

Графики этим скриптом не рисуются — для презентационных PNG используется
`scripts/make_presentation_charts.py`, который умеет читать i2i‑файлы.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from train_ltr_full import (
    LTR_FEATURE_NAMES,
    MODELS_DIR,
    REPORTS_DIR,
    build_dataset,
    fit_retriever,
    load_articles,
    load_coview_aligned,
    load_embeddings_memmap,
    load_logs,
    time_split,
)

# ---------------------------------------------------------------------------
# i2i entity_types — только эти блоки являются item‑to‑item по постановке.
# `ml_personal-*` — это user→items (персонализация на user history),
# `popularity-block-*` — глобальный топ просмотров без смысловой связи с источником.
# Оба исключаем из оценки.
# ---------------------------------------------------------------------------
I2I_ENTITY_TYPES = {
    "article.ml_what-else-mi-pisali-block-recommendation",
    "article.what-else-mi-pisali-block-recommendation",
}


def predict_lgbm(model, X: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    return np.asarray(model.predict(df), dtype=np.float32)


def predict_catboost(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=np.float32)


def metrics_at_k(y, score, group, ks=(1, 3, 5, 6, 10), min_group_size: int = 2):
    """Возвращает (metrics_dict, n_groups_evaluated).

    Считает recall@K, nDCG@K, MRR@K в каждой группе и усредняет.
    Пропускает группы с size < min_group_size и группы без позитивов.
    """

    out: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
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
    n = len(out["mrr@3"])
    return {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}, n


def per_entity_table(y, score, group, et_per_group: pd.Series, k: int = 6, min_group_size: int = 2):
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
    print("[load] articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)
    print(f"      articles: {art.df.shape[0]:,}")

    print("[load] LTR model bundle…")
    bundle = pickle.loads((MODELS_DIR / "ltr_lgbm.pkl").read_bytes())
    model_lgbm = bundle["model"]
    propensity = bundle.get("propensity", {})

    print("[load] coview index…")
    _, _, coview_pair_df = load_coview_aligned(art)

    print("[load] logs…")
    logs = load_logs()
    _, test_logs, cut = time_split(logs, train_quantile=0.8)
    del logs

    n_test_full = len(test_logs)
    test_logs = test_logs[test_logs["entity_type"].astype(str).isin(I2I_ENTITY_TYPES)].copy()
    n_test_i2i = len(test_logs)
    print(f"[filter] test rows: {n_test_full:,} → {n_test_i2i:,} (i2i only, {n_test_i2i/max(n_test_full,1)*100:.1f}%)")
    if n_test_i2i == 0:
        raise RuntimeError("после фильтра по I2I_ENTITY_TYPES не осталось строк в test")

    print("[build] test dataset (i2i)…")
    Xte, yte, gte, _, _, et_te = build_dataset(
        art, test_logs, max_groups=150_000, coview_pair_df=coview_pair_df,
    )
    del test_logs
    print(f"      test rows: {len(yte):,}  groups: {len(gte):,}  positives: {int(yte.sum()):,}")

    print("[predict] scoring…")
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

    ks = (1, 3, 5, 6, 10)
    print("[eval] computing metrics (i2i only)…")
    res = {}
    res["lgbm"], n_eval = metrics_at_k(yte, s_lgbm, gte, ks=ks)
    res["baseline_sim"], _ = metrics_at_k(yte, s_baseline, gte, ks=ks)
    res["random"], _ = metrics_at_k(yte, s_random, gte, ks=ks)
    if s_cb is not None:
        res["catboost"], _ = metrics_at_k(yte, s_cb, gte, ks=ks)
    print(f"  groups evaluated: {n_eval:,}")

    # --- per-entity таблица (для контроля; на i2i таких типа всего два)
    pe_lgbm = per_entity_table(yte, s_lgbm, gte, et_te)
    pe_base = per_entity_table(yte, s_baseline, gte, et_te)
    per_entity = pe_lgbm.merge(
        pe_base.rename(columns={"ndcg@6": "ndcg@6_baseline", "mrr@6": "mrr@6_baseline"}).drop(columns=["n_groups"]),
        on="entity_type", how="left",
    )

    # --- сохранение ---------------------------------------------------------
    extra_path = REPORTS_DIR / "ltr_extra_metrics_i2i.json"
    extra_path.write_text(
        json.dumps({
            "groups_evaluated": n_eval,
            "i2i_entity_types": sorted(I2I_ENTITY_TYPES),
            "test_rows_total": int(n_test_full),
            "test_rows_i2i": int(n_test_i2i),
            "metrics": res,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] {extra_path}")

    full_path = REPORTS_DIR / "ltr_full_metrics_i2i.json"
    overall = {name: {
        f"ndcg@{k}": vals[f"ndcg@{k}"] for k in (3, 6, 10)
    } | {
        f"mrr@{k}": vals[f"mrr@{k}"] for k in (3, 6, 10)
    } for name, vals in res.items()}
    full_path.write_text(
        json.dumps({
            "cut": str(cut),
            "i2i_entity_types": sorted(I2I_ENTITY_TYPES),
            "test_rows_total": int(n_test_full),
            "test_rows_i2i": int(n_test_i2i),
            "test_groups_evaluated": n_eval,
            "test_positives": int(yte.sum()),
            "propensity": {int(k): float(v) for k, v in propensity.items()},
            "overall": overall,
            "per_entity": per_entity.to_dict(orient="records"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[save] {full_path}")

    # --- человекочитаемый отчёт в stdout
    print()
    print("=" * 80)
    print(f"i2i evaluation summary  (groups: {n_eval:,}; test rows: {n_test_i2i:,})")
    print("=" * 80)
    cols = ["recall@1", "ndcg@3", "ndcg@6", "mrr@3", "mrr@6"]
    print(f"  model            " + "  ".join(f"{c:>9}" for c in cols))
    for name, vals in res.items():
        print(f"  {name:<16} " + "  ".join(f"{vals[c]:>9.4f}" for c in cols))
    print()
    print("delta LTR (lgbm) − baseline_sim:")
    for c in cols:
        d = res["lgbm"][c] - res["baseline_sim"][c]
        rel = d / max(res["baseline_sim"][c], 1e-9) * 100
        print(f"  {c:<10}  +{d:.4f}  ({rel:+.1f}%)")


if __name__ == "__main__":
    main()
