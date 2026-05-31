"""Обучает LightGBM БЕЗ coview-фич и синтетических coview-групп, затем
оценивает на чистой i2i подвыборке (как `eval_ltr_i2i.py`). Используется
для честного A/B-сравнения «без coview vs с coview» на одной и той же
выборке.

Что делает:
  1. Грузит логи и articles, как `train_ltr_full.main`.
  2. Строит train без coview-пар (coview_score/coview_rank = missing-константа).
  3. НЕ добавляет синтетические coview-группы.
  4. Тренирует LightGBM с теми же гиперпараметрами.
  5. Оценивает на i2i подвыборке test (фильтр по I2I_ENTITY_TYPES).
  6. Сравнивает с метриками из `reports/ltr_extra_metrics_i2i.json` (с coview).
  7. Сохраняет:
       - models/ltr_lgbm_no_coview.pkl
       - reports/ltr_extra_metrics_no_coview_i2i.json
       - reports/coview_uplift_i2i.json (diff: с coview – без coview)
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval_ltr_i2i import I2I_ENTITY_TYPES, metrics_at_k
from train_ltr_full import (
    LTR_FEATURE_NAMES,
    MODELS_DIR,
    REPORTS_DIR,
    build_dataset,
    estimate_position_propensity,
    fit_retriever,
    load_articles,
    load_embeddings_memmap,
    load_logs,
    time_split,
    train_lgbm,
)


def predict_lgbm(model, X: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    return np.asarray(model.predict(df), dtype=np.float32)


def main() -> None:
    print("[1/6] loading articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)

    print("[2/6] loading logs…")
    logs = load_logs()
    train_logs, test_logs, cut = time_split(logs, train_quantile=0.8)
    del logs

    print("[3/6] propensity (train)…")
    propensity = estimate_position_propensity(train_logs)

    print("[4/6] building TRAIN dataset WITHOUT coview…")
    t0 = time.time()
    Xtr, ytr, gtr, wtr, _, _ = build_dataset(
        art, train_logs, propensity=propensity, max_groups=600_000,
        coview_pair_df=None,
    )
    del train_logs
    print(f"      train rows: {len(ytr):,}  groups: {len(gtr):,}  positives: {int(ytr.sum()):,}  ({time.time()-t0:.0f}s)")
    print("      [note] coview_score / coview_rank всегда missing → как фичи неактивны")

    print("[5/6] training LightGBM (no coview, no synth)…")
    t0 = time.time()
    model = train_lgbm(Xtr, ytr, gtr, sample_weight=wtr)
    print(f"      lgbm trained in {time.time()-t0:.0f}s")

    out_model = MODELS_DIR / "ltr_lgbm_no_coview.pkl"
    with open(out_model, "wb") as f:
        pickle.dump({"model": model, "feature_names": LTR_FEATURE_NAMES, "propensity": propensity}, f)
    print(f"      saved {out_model}")

    print("[6/6] building TEST dataset (i2i only, WITHOUT coview)…")
    n_full = len(test_logs)
    test_logs = test_logs[test_logs["entity_type"].astype(str).isin(I2I_ENTITY_TYPES)].copy()
    n_i2i = len(test_logs)
    print(f"      filter: {n_full:,} → {n_i2i:,} ({n_i2i/max(n_full,1)*100:.1f}%)")

    Xte, yte, gte, _, _, _ = build_dataset(
        art, test_logs, max_groups=150_000, coview_pair_df=None,
    )
    del test_logs
    print(f"      test rows: {len(yte):,}  groups: {len(gte):,}  positives: {int(yte.sum()):,}")

    print("[eval] computing metrics on i2i (no coview)…")
    s = predict_lgbm(model, Xte)
    s_baseline = Xte[:, 0]
    rng = np.random.default_rng(123)
    s_random = rng.standard_normal(len(yte)).astype(np.float32)

    ks = (1, 3, 5, 6, 10)
    res = {}
    res["lgbm"], n_eval = metrics_at_k(yte, s, gte, ks=ks)
    res["baseline_sim"], _ = metrics_at_k(yte, s_baseline, gte, ks=ks)
    res["random"], _ = metrics_at_k(yte, s_random, gte, ks=ks)

    # CatBoost — переобучать не обязательно для основного сравнения, пропускаем
    print(f"  groups evaluated: {n_eval:,}")

    out_metrics = REPORTS_DIR / "ltr_extra_metrics_no_coview_i2i.json"
    out_metrics.write_text(json.dumps({
        "mode": "no_coview_i2i",
        "groups_evaluated": n_eval,
        "test_rows_total": n_full,
        "test_rows_i2i": n_i2i,
        "metrics": res,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      saved {out_metrics}")

    # --- сравнение с coview-моделью
    coview_path = REPORTS_DIR / "ltr_extra_metrics_i2i.json"
    if coview_path.exists():
        coview_res = json.loads(coview_path.read_text())["metrics"]
        cmp_rows = []
        keys_to_show = ["recall@1", "recall@3", "recall@6", "ndcg@3", "ndcg@6", "ndcg@10",
                        "mrr@3", "mrr@6", "mrr@10"]
        print()
        print("=" * 70)
        print("LightGBM: no_coview vs with_coview (i2i)")
        print("=" * 70)
        print(f"  {'metric':<10}  {'no coview':>10}  {'with coview':>12}  {'Δ pp':>9}  {'Δ rel':>8}")
        for k in keys_to_show:
            a = res["lgbm"][k]
            b = coview_res["lgbm"][k]
            d = b - a
            rel = d / max(a, 1e-9) * 100
            print(f"  {k:<10}  {a:>10.4f}  {b:>12.4f}  {d:>+9.4f}  {rel:>+7.2f}%")
            cmp_rows.append({"metric": k, "no_coview": a, "with_coview": b, "delta_pp": d, "delta_rel_pct": rel})

        (REPORTS_DIR / "coview_uplift_i2i.json").write_text(
            json.dumps({"groups_evaluated": n_eval, "comparison": cmp_rows},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nsaved {REPORTS_DIR / 'coview_uplift_i2i.json'}")
    else:
        print(f"  (skip diff: {coview_path} not found)")


if __name__ == "__main__":
    main()
