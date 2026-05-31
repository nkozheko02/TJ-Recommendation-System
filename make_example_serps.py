"""Сгенерировать примеры выдачи (нашa LTR vs baseline_sim) для нескольких
статей-источников и сохранить таблицы для приложения к диплому.

Делает:
- для каждой выбранной статьи строит ТОП-K кандидатов из эмбеддингов
  (≈ 80 ближайших соседей);
- считает фичи и скорит обученной LightGBM-моделью;
- получает упорядочения у двух систем (LTR и baseline по sim);
- сохраняет таблицы в reports/examples/serp_<id>.csv;
- собирает мини-таблицу для диплома: top-6 у каждой системы.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train_ltr_full import (
    COVIEW_RANK_MISSING,
    COVIEW_SCORE_MISSING,
    LTR_FEATURE_NAMES,
    MODELS_DIR,
    REPORTS_DIR,
    fit_retriever,
    load_articles,
    load_coview_aligned,
    load_embeddings_memmap,
    prepare_feature_cache,
    safe_str,
)

EXAMPLES_DIR = REPORTS_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)


def topk_candidates(art, source_id: str, k_pool: int = 80) -> tuple[int, np.ndarray, np.ndarray]:
    src_idx = art.article_id_to_row.get(source_id)
    if src_idx is None:
        raise KeyError(f"article_id {source_id} not in retriever index")
    src_vec = art.X[src_idx : src_idx + 1]
    dist, idx = art.nn.kneighbors(src_vec, n_neighbors=k_pool + 1)
    cand_idx = idx[0][1:]  # без самого себя
    sim = 1.0 - dist[0][1:]
    return int(src_idx), cand_idx.astype(np.int64), sim.astype(np.float32)


def build_pair_features(
    art,
    cache: dict,
    src_idx: int,
    cand_idx: np.ndarray,
    sim: np.ndarray,
    *,
    coview_neigh_idx: dict | None = None,
    coview_neigh_score: dict | None = None,
) -> np.ndarray:
    n = len(cand_idx)

    src_views = float(cache["log_views"][src_idx])
    src_like = float(cache["like_rate"][src_idx])
    src_fresh = float(cache["fresh"][src_idx])
    src_age = float(cache["age_days"][src_idx])
    src_author = int(cache["author_code"][src_idx])
    src_dept = int(cache["dept_code"][src_idx])
    src_rub = int(cache["rub_code"][src_idx])

    cand_views = cache["log_views"][cand_idx]
    cand_log_comments = cache["log_comments"][cand_idx]
    cand_like = cache["like_rate"][cand_idx]
    cand_comment = cache["comment_rate"][cand_idx]
    cand_fav = cache["fav_rate"][cand_idx]
    cand_fresh = cache["fresh"][cand_idx]
    cand_age = cache["age_days"][cand_idx]
    cand_author = cache["author_code"][cand_idx]
    cand_dept = cache["dept_code"][cand_idx]
    cand_rub = cache["rub_code"][cand_idx]

    same_author = (cand_author == src_author).astype(np.float32)
    abs_age_diff = np.abs(cand_age - src_age).astype(np.float32)
    same_dept = (cand_dept == src_dept).astype(np.float32)
    same_rub = (cand_rub == src_rub).astype(np.float32)

    pos_f = np.full(n, 1.0, dtype=np.float32)

    coview_score = np.full(n, COVIEW_SCORE_MISSING, dtype=np.float32)
    coview_rank = np.full(n, COVIEW_RANK_MISSING, dtype=np.float32)
    if coview_neigh_idx is not None and src_idx in coview_neigh_idx:
        neigh = coview_neigh_idx[src_idx]
        scores = coview_neigh_score[src_idx]
        cand_to_pos = {int(c): r for r, c in enumerate(neigh.tolist())}
        for j, cand in enumerate(cand_idx.tolist()):
            r = cand_to_pos.get(int(cand))
            if r is not None:
                coview_score[j] = float(scores[r])
                coview_rank[j] = float(r)

    X = np.column_stack([
        sim,
        cand_views,
        cand_like,
        cand_comment,
        cand_log_comments,
        cand_fav,
        cand_fresh,
        same_author,
        abs_age_diff,
        same_dept,
        same_rub,
        np.full(n, src_views, dtype=np.float32),
        np.full(n, src_like, dtype=np.float32),
        np.full(n, src_fresh, dtype=np.float32),
        pos_f,
        coview_score,
        coview_rank,
    ]).astype(np.float32)
    return X


def head_short(s: str, n: int = 70) -> str:
    s = safe_str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def make_serp_table(
    art,
    cache: dict,
    model,
    source_id: str,
    *,
    k_show: int = 8,
    pool: int = 80,
    coview_neigh_idx: dict | None = None,
    coview_neigh_score: dict | None = None,
) -> pd.DataFrame:
    src_idx, cand_idx, sim = topk_candidates(art, source_id, k_pool=pool)
    X = build_pair_features(
        art, cache, src_idx, cand_idx, sim,
        coview_neigh_idx=coview_neigh_idx,
        coview_neigh_score=coview_neigh_score,
    )

    df_inp = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    ltr_score = model.predict(df_inp)

    sim_order = np.argsort(-sim)[:k_show]
    ltr_order = np.argsort(-ltr_score)[:k_show]

    rows = []
    for rank, ord_arr, name in [(None, sim_order, "baseline_sim"), (None, ltr_order, "LTR (LightGBM)")]:
        for r, j in enumerate(ord_arr, start=1):
            row = art.df.iloc[int(cand_idx[j])]
            rows.append({
                "system": name,
                "rank": r,
                "title": head_short(row.get("article_base__title", ""), 70),
                "rubric": safe_str(row.get("article_base__rubric", "")),
                "author": safe_str(row.get("article_base__author_name", "")),
                "age_days": int(row.get("article_dates__days_since_published", 0) or 0),
                "views": int(row.get("article_stats__stats_views", 0) or 0),
                "sim": round(float(sim[j]), 3),
                "ltr_score": round(float(ltr_score[j]), 3),
            })
    return pd.DataFrame(rows)


def find_interesting_sources(art, k: int = 6) -> list[str]:
    """Подобрать пару статей-источников: одну «канонически популярную»,
    одну посвежее. Берём конкретные id, если они существуют, иначе
    выбираем случайные с большим числом просмотров.
    """

    df = art.df.copy()
    df["views"] = pd.to_numeric(df.get("article_stats__stats_views", 0), errors="coerce").fillna(0)
    df["age"] = pd.to_numeric(df.get("article_dates__days_since_published", 365), errors="coerce").fillna(365)

    pool_old = df[df["age"].between(60, 365)].sort_values("views", ascending=False).head(50)
    pool_new = df[df["age"].between(7, 60)].sort_values("views", ascending=False).head(50)
    rng = np.random.default_rng(7)
    chosen = []
    if len(pool_old):
        chosen.append(pool_old.iloc[int(rng.integers(0, len(pool_old)))]["article_id"])
    if len(pool_new):
        chosen.append(pool_new.iloc[int(rng.integers(0, len(pool_new)))]["article_id"])
    return [safe_str(x) for x in chosen[:k]]


def main() -> None:
    print("[load] articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)

    print("[load] LTR model…")
    bundle = pickle.loads((MODELS_DIR / "ltr_lgbm.pkl").read_bytes())
    model = bundle["model"]

    cache = prepare_feature_cache(art)

    print("[load] coview index…")
    coview_neigh_idx, coview_neigh_score, _ = load_coview_aligned(art)

    sources = find_interesting_sources(art)
    print("sources:", sources)

    out_summary = []
    for sid in sources:
        try:
            t = make_serp_table(
                art, cache, model, sid, k_show=8, pool=80,
                coview_neigh_idx=coview_neigh_idx,
                coview_neigh_score=coview_neigh_score,
            )
        except KeyError as e:
            print("skip:", e)
            continue
        title = head_short(art.df.set_index("article_id").loc[sid, "article_base__title"], 70)
        path_csv = EXAMPLES_DIR / f"serp_{sid}.csv"
        t.to_csv(path_csv, index=False)
        print("  saved:", path_csv)
        out_summary.append({"source_id": sid, "source_title": title, "table_path": str(path_csv)})

    (EXAMPLES_DIR / "summary.json").write_text(
        json.dumps(out_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("done.")


if __name__ == "__main__":
    main()
