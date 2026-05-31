"""Полный пайплайн обучения LTR-модели рекомендательной системы Т-Ж.

Что делает:
- читает все логи показов карусели (с time-split);
- собирает дата-сет LambdaRank по группам показов
  (group = visit_id | source_article_id | entity_type | day);
- учитывает propensity по позиции (debias);
- обучает LightGBM (LambdaRank) и CatBoost (YetiRank);
- оценивает на test-окне (nDCG@K, MRR@K, Recall@K)
  с разрезами по entity_type;
- сохраняет модели в `models/` и метрики/графики в `reports/`.

Запуск:
    .venv/bin/python train_ltr_full.py
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import ndcg_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
ARTICLE_CSV = ROOT / "tj_article.csv"
EMB_DIR = ROOT / "embeddings_cache"
LOGS_PARQUET = Path("/Users/nkozheko/Downloads/tj_session_w_target_full.parquet")
COVIEW_PATH = ROOT / "coview_cache" / "coview_index.pkl.gz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

EVAL_KS = (3, 6, 10)
RANDOM_SEED = 42

# Когда у пары (a, b) нет записи в coview-индексе, считаем что это
# «никогда не наблюдалось вместе»: log p ≈ -∞. Заменяем на этот floor
# (примерно ниже минимального score из реального индекса).
COVIEW_SCORE_MISSING = -10.0
COVIEW_RANK_MISSING = 200  # больше типичного длинного coview-хвоста

# ---------------------------------------------------------------------------
# helpers (copy of relevant notebook utils, fixed and self-contained)
# ---------------------------------------------------------------------------


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def quality_prior(row: pd.Series) -> float:
    like_rate = row.get("article_stats__like_rate", np.nan)
    comment_rate = row.get("article_stats__comment_rate", np.nan)
    favs = row.get("article_stats__stats_favorites", np.nan)
    views = row.get("article_stats__stats_views", np.nan)
    lr = float(like_rate) if pd.notna(like_rate) else 0.0
    cr = float(comment_rate) if pd.notna(comment_rate) else 0.0
    fav_rate = (
        float(favs) / max(float(views), 1.0)
        if pd.notna(favs) and pd.notna(views)
        else 0.0
    )
    return float(_sigmoid(12.0 * (0.7 * lr + 0.3 * fav_rate) + 4.0 * cr))


def trending_prior(row: pd.Series) -> float:
    views = row.get("article_stats__stats_views", np.nan)
    days = row.get("article_dates__days_since_published", np.nan)
    v = float(views) if pd.notna(views) else 0.0
    d = float(days) if pd.notna(days) else 365.0
    return float(math.log1p(v) / math.sqrt(d + 1.0))


def freshness(row: pd.Series) -> float:
    days = row.get("article_dates__days_since_published", np.nan)
    d = float(days) if pd.notna(days) else 365.0
    return float(1.0 / (1.0 + d / 30.0))


def load_articles(path: Path = ARTICLE_CSV) -> pd.DataFrame:
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(path, sep=";", encoding="utf-8", encoding_errors="replace", low_memory=False)

    df["article_id"] = df["article_id"].map(safe_str)
    for col in [
        "article_base__title",
        "article_base__department",
        "article_base__rubric",
        "article_base__author_id",
        "article_base__author_name",
    ]:
        if col in df.columns:
            df[col] = df[col].map(safe_str)

    numeric_prefixes = ("article_stats__", "article_dates__", "article_author__")
    for col in df.columns:
        if not col.startswith(numeric_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


def load_embeddings_memmap(out_dir: Path = EMB_DIR) -> tuple[list[str], np.ndarray]:
    n_str, dim_str = (
        (out_dir / "shape.txt").read_text(encoding="utf-8").strip().split(",")
    )
    n, dim = int(n_str), int(dim_str)
    ids = [
        line.strip()
        for line in (out_dir / "ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    X = np.memmap(out_dir / "embeddings.f32", dtype="float32", mode="r", shape=(n, dim))
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = (X / norms).astype(np.float32)
    return ids, X


@dataclass(frozen=True)
class Artifacts:
    df: pd.DataFrame
    article_id_to_row: dict[str, int]
    nn: NearestNeighbors
    X: np.ndarray


def fit_retriever(df: pd.DataFrame, X_ids: list[str], X: np.ndarray) -> Artifacts:
    id_to_emb_row = {aid: i for i, aid in enumerate(X_ids) if aid}
    keep_rows: list[int] = []
    emb_rows: list[int] = []
    for i, aid in enumerate(df["article_id"].map(safe_str).tolist()):
        j = id_to_emb_row.get(aid)
        if j is None:
            continue
        keep_rows.append(i)
        emb_rows.append(j)
    df2 = df.iloc[keep_rows].copy().reset_index(drop=True)
    X2 = X[np.asarray(emb_rows, dtype=np.int64)]
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X2)
    article_id_to_row = {aid: i for i, aid in enumerate(df2["article_id"].tolist()) if aid}
    return Artifacts(df=df2, article_id_to_row=article_id_to_row, nn=nn, X=X2)


# ---------------------------------------------------------------------------
# logs loading / time-split
# ---------------------------------------------------------------------------

LOG_COLS = [
    "visit_id",
    "hit_dttm",
    "source_article_id",
    "target_article_id",
    "entity_type",
    "source_type",
    "article_position",
    "target",
]


def _hash_uint64(*parts: object) -> int:
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little", signed=False)


def _hash_series_uint64(*series: pd.Series) -> np.ndarray:
    """Векторизованный хеш по нескольким сериям. Используем pandas .factorize
    для каждой колонки и объединяем коды через mix-функцию.
    Так значительно быстрее и легче по памяти, чем построчный hashlib.
    """

    arr = None
    for s in series:
        codes, _ = pd.factorize(s, sort=False)
        codes = codes.astype(np.uint64) + 1
        if arr is None:
            arr = codes
        else:
            arr = (arr * np.uint64(1_000_003)) ^ codes
    return arr


def load_logs(parquet_path: Path = LOGS_PARQUET) -> pd.DataFrame:
    """Читает parquet потоково, отфильтровывает шум, держит только компактные колонки.

    Возвращает DataFrame со столбцами:
        impr_key        (uint64)   — хеш (visit, source, entity_type, day)
        source_article_id (object) — исходный uuid
        target_article_id (object) — uuid кандидата
        entity_type     (category)
        article_position (int8)
        target          (int8)
        day_int         (int32) — дни с эпохи (для time-split)
        hit_dttm        (datetime64[ns, UTC]) — для построения графика split
    """

    print(f"[logs] reading {parquet_path}")
    t = time.time()
    dataset = ds.dataset(parquet_path, format="parquet")
    cols = [c for c in LOG_COLS if c in dataset.schema.names]
    scan = dataset.scanner(columns=cols, batch_size=400_000)

    chunks = []
    raw_rows = 0
    kept_rows = 0
    for i, batch in enumerate(scan.to_batches()):
        pdf = batch.to_pandas(strings_to_categorical=False)
        raw_rows += len(pdf)
        pdf["entity_type"] = pdf["entity_type"].fillna("").astype(str)
        pdf["source_type"] = pdf["source_type"].fillna("").astype(str)
        pdf = pdf[(pdf["entity_type"] != "") & (pdf["source_type"] != "coview")]
        if len(pdf) == 0:
            continue

        pdf["source_article_id"] = pdf["source_article_id"].map(safe_str)
        pdf["target_article_id"] = pdf["target_article_id"].map(safe_str)
        pdf["visit_id"] = pdf["visit_id"].map(safe_str)
        pdf = pdf[(pdf["visit_id"] != "") & (pdf["source_article_id"] != "") & (pdf["target_article_id"] != "")]

        ts = pd.to_datetime(pdf["hit_dttm"], errors="coerce", utc=True)
        day = ts.dt.date.astype(str)
        impr_key = _hash_series_uint64(
            pdf["visit_id"].astype(str),
            pdf["source_article_id"].astype(str),
            pdf["entity_type"].astype(str),
            day,
        )

        out = pd.DataFrame({
            "impr_key": impr_key,
            "source_article_id": pdf["source_article_id"].values,
            "target_article_id": pdf["target_article_id"].values,
            "entity_type": pdf["entity_type"].astype("category").values,
            "article_position": pd.to_numeric(pdf.get("article_position", -1), errors="coerce").fillna(-1).astype(np.int8).values,
            "target": pd.to_numeric(pdf["target"], errors="coerce").fillna(0).astype(np.int8).values,
            "hit_dttm": ts.reset_index(drop=True),
        })
        chunks.append(out)
        kept_rows += len(out)
        if (i + 1) % 10 == 0:
            print(f"  [logs]   batch {i+1:>3}: kept={kept_rows:,} / raw={raw_rows:,}  elapsed={time.time()-t:.1f}s")

    res = pd.concat(chunks, ignore_index=True)
    res["entity_type"] = res["entity_type"].astype("category")
    print(f"[logs] kept rows after filter: {len(res):,}  ({time.time()-t:.1f}s)")
    return res


def time_split(df: pd.DataFrame, *, train_quantile: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ts = df["hit_dttm"]
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    cut = pd.Timestamp(ts.quantile(train_quantile))
    train = df[ts < cut].copy()
    test = df[ts >= cut].copy()
    print(f"[split] cut={cut}  train={len(train):,}  test={len(test):,}")
    return train, test, cut


# ---------------------------------------------------------------------------
# feature engineering — единственная точка истины для inference тоже
# ---------------------------------------------------------------------------

LTR_FEATURE_NAMES = [
    "sim",
    "cand_log_views",
    "cand_like_rate",
    "cand_comment_rate",
    "cand_log_comments",
    "cand_fav_rate",
    "cand_fresh",
    "same_author",
    "abs_age_diff_days",
    "same_dept",
    "same_rubric",
    "src_log_views",
    "src_like_rate",
    "src_fresh",
    "pos",
    "coview_score",
    "coview_rank",
]


def _log1p_series(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32).values.copy()
    v[v < 0] = 0
    return np.log1p(v).astype(np.float32)


def _rate_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32).values.copy()


def prepare_feature_cache(art: Artifacts) -> dict[str, np.ndarray]:
    """Все per-article фичи кэшируются в numpy — индексирование O(1), без копий."""

    d = art.df
    views = _log1p_series(d.get("article_stats__stats_views", 0.0))
    comments_log = _log1p_series(d.get("article_stats__stats_comments", 0.0))
    like_rate = _rate_series(d.get("article_stats__like_rate", 0.0))
    comment_rate = _rate_series(d.get("article_stats__comment_rate", 0.0))
    favs = pd.to_numeric(d.get("article_stats__stats_favorites", 0.0), errors="coerce").fillna(0.0).astype(np.float32).values
    raw_views = pd.to_numeric(d.get("article_stats__stats_views", 0.0), errors="coerce").fillna(0.0).astype(np.float32).values
    fav_rate = (favs / np.maximum(raw_views, 1.0)).astype(np.float32)
    age_days = pd.to_numeric(d.get("article_dates__days_since_published", 365.0), errors="coerce").fillna(365.0).astype(np.float32).values
    fresh = (1.0 / (1.0 + age_days / 30.0)).astype(np.float32)

    def _codes(col: str) -> np.ndarray:
        s = d.get(col, "")
        if isinstance(s, str):
            return np.zeros(len(d), dtype=np.int32)
        codes, _ = pd.factorize(s.map(safe_str), sort=False)
        return codes.astype(np.int32)

    return {
        "log_views": views,
        "log_comments": comments_log,
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "fav_rate": fav_rate,
        "fresh": fresh,
        "age_days": age_days,
        "author_code": _codes("article_base__author_id"),
        "dept_code": _codes("article_base__department"),
        "rub_code": _codes("article_base__rubric"),
    }


# ---------------------------------------------------------------------------
# coview index loading
# ---------------------------------------------------------------------------


def load_coview_aligned(
    art: Artifacts,
    path: Path = COVIEW_PATH,
    *,
    top_k: int = 100,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    """Загружает coview-индекс из pkl.gz и выравнивает по row_idx из art.

    Возвращает три структуры:
        neighbors_idx[src_idx]   -> np.ndarray cand_idx, отсортированы по убыванию score
        neighbors_score[src_idx] -> np.ndarray scores (тех же длин)
        pair_df                  -> DataFrame[src_idx, cand_idx, coview_score, coview_rank]
                                    для быстрого merge на этапе фичей.
    """

    import gzip
    import pickle

    print(f"[coview] reading {path}")
    t = time.time()
    with gzip.open(path, "rb") as f:
        raw: dict[str, list[tuple[str, float]]] = pickle.load(f)

    id_to_idx = art.article_id_to_row
    neighbors_idx: dict[int, np.ndarray] = {}
    neighbors_score: dict[int, np.ndarray] = {}

    rows_src: list[int] = []
    rows_cand: list[int] = []
    rows_score: list[float] = []
    rows_rank: list[int] = []

    for src_id, neigh in raw.items():
        src_idx = id_to_idx.get(src_id)
        if src_idx is None:
            continue
        cand_idx_list: list[int] = []
        score_list: list[float] = []
        for cand_id, score in neigh:
            ci = id_to_idx.get(cand_id)
            if ci is None:
                continue
            cand_idx_list.append(ci)
            score_list.append(float(score))
            if len(cand_idx_list) >= top_k:
                break
        if not cand_idx_list:
            continue
        ci_arr = np.asarray(cand_idx_list, dtype=np.int64)
        sc_arr = np.asarray(score_list, dtype=np.float32)
        neighbors_idx[src_idx] = ci_arr
        neighbors_score[src_idx] = sc_arr
        for r, (ci, sc) in enumerate(zip(cand_idx_list, score_list)):
            rows_src.append(src_idx)
            rows_cand.append(ci)
            rows_score.append(sc)
            rows_rank.append(r)

    pair_df = pd.DataFrame({
        "src_idx": np.asarray(rows_src, dtype=np.int64),
        "tgt_idx": np.asarray(rows_cand, dtype=np.int64),
        "coview_score": np.asarray(rows_score, dtype=np.float32),
        "coview_rank": np.asarray(rows_rank, dtype=np.int32),
    })
    print(
        f"[coview] aligned sources: {len(neighbors_idx):,}  pairs: {len(pair_df):,}"
        f"  ({time.time()-t:.1f}s)"
    )
    return neighbors_idx, neighbors_score, pair_df


def estimate_position_propensity(logs: pd.DataFrame, *, alpha: float = 1.0, beta: float = 1.0) -> dict[int, float]:
    g = logs.groupby("article_position", observed=True, sort=False)["target"]
    out: dict[int, float] = {}
    for pos, vals in g:
        n = len(vals)
        k = int(vals.sum())
        out[int(pos)] = (k + alpha) / (n + alpha + beta)
    return out


def build_dataset(
    art: Artifacts,
    logs: pd.DataFrame,
    *,
    propensity: dict[int, float] | None = None,
    propensity_floor: float = 0.02,
    min_group_size: int = 2,
    max_groups: int | None = None,
    seed: int = RANDOM_SEED,
    coview_pair_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, list[int], pd.Series]:
    """Возвращает (X, y, group_sizes, w, group_keys, entity_type_per_group)."""

    rng = np.random.default_rng(seed)

    df = logs

    # оставляем только пары, которые есть в aligned df
    src_map = pd.Series(art.article_id_to_row, name="src_idx")
    tgt_map = pd.Series(art.article_id_to_row, name="tgt_idx")
    df = df.assign(
        src_idx=df["source_article_id"].map(src_map),
        tgt_idx=df["target_article_id"].map(tgt_map),
    )
    df = df.dropna(subset=["src_idx", "tgt_idx"])
    df["src_idx"] = df["src_idx"].astype(np.int64)
    df["tgt_idx"] = df["tgt_idx"].astype(np.int64)

    sizes = df.groupby("impr_key", sort=False).size()
    big = sizes[sizes >= min_group_size].index
    df = df[df["impr_key"].isin(big)]
    print(f"[dataset] groups>={min_group_size}: {len(big):,}  rows: {len(df):,}")

    if max_groups is not None and len(big) > max_groups:
        sel_idx = rng.choice(len(big), size=max_groups, replace=False)
        keep_keys = set(big.values[sel_idx].tolist())
        df = df[df["impr_key"].isin(keep_keys)]
        print(f"[dataset] sampled to {max_groups:,} groups → rows: {len(df):,}")

    df = df.sort_values("impr_key", kind="stable").reset_index(drop=True)
    n_groups_final = df["impr_key"].nunique()
    print(f"[dataset] sorted, final rows: {len(df):,}  unique groups: {n_groups_final:,}")

    cache = prepare_feature_cache(art)

    src_idx_arr = df["src_idx"].values
    cand_idx_arr = df["tgt_idx"].values

    src_views = cache["log_views"][src_idx_arr]
    src_like = cache["like_rate"][src_idx_arr]
    src_fresh = cache["fresh"][src_idx_arr]
    src_age = cache["age_days"][src_idx_arr]
    src_author = cache["author_code"][src_idx_arr]
    src_dept = cache["dept_code"][src_idx_arr]
    src_rub = cache["rub_code"][src_idx_arr]

    cand_views = cache["log_views"][cand_idx_arr]
    cand_log_comments = cache["log_comments"][cand_idx_arr]
    cand_like = cache["like_rate"][cand_idx_arr]
    cand_comment = cache["comment_rate"][cand_idx_arr]
    cand_fav = cache["fav_rate"][cand_idx_arr]
    cand_fresh = cache["fresh"][cand_idx_arr]
    cand_age = cache["age_days"][cand_idx_arr]
    cand_author = cache["author_code"][cand_idx_arr]
    cand_dept = cache["dept_code"][cand_idx_arr]
    cand_rub = cache["rub_code"][cand_idx_arr]

    same_author = (cand_author == src_author).astype(np.float32)
    abs_age_diff = np.abs(cand_age - src_age).astype(np.float32)
    same_dept = (cand_dept == src_dept).astype(np.float32)
    same_rub = (cand_rub == src_rub).astype(np.float32)

    # similarity по эмбеддингам — батчами, чтобы не материализовать 7M×1024 в RAM
    sim = np.empty(len(src_idx_arr), dtype=np.float32)
    chunk = 80_000
    for i in range(0, len(src_idx_arr), chunk):
        sl = slice(i, i + chunk)
        vs = art.X[src_idx_arr[sl]]
        vc = art.X[cand_idx_arr[sl]]
        sim[sl] = np.einsum("ij,ij->i", vs, vc, optimize=True)
        del vs, vc

    pos = df["article_position"].astype(np.int32).values
    pos_f = pos.astype(np.float32)

    # coview-фичи: merge с парным индексом (если он передан)
    if coview_pair_df is not None and len(coview_pair_df):
        merged = df[["src_idx", "tgt_idx"]].merge(
            coview_pair_df, on=["src_idx", "tgt_idx"], how="left"
        )
        coview_score = merged["coview_score"].fillna(COVIEW_SCORE_MISSING).astype(np.float32).values
        coview_rank = merged["coview_rank"].fillna(COVIEW_RANK_MISSING).astype(np.float32).values
    else:
        coview_score = np.full(len(df), COVIEW_SCORE_MISSING, dtype=np.float32)
        coview_rank = np.full(len(df), COVIEW_RANK_MISSING, dtype=np.float32)

    X = np.column_stack(
        [
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
            src_views,
            src_like,
            src_fresh,
            pos_f,
            coview_score,
            coview_rank,
        ]
    ).astype(np.float32)

    y = df["target"].astype(np.float32).values

    if propensity:
        p = np.array([propensity.get(int(pp), propensity_floor) for pp in pos], dtype=np.float32)
        p = np.clip(p, propensity_floor, 1.0)
        w = (1.0 / p).astype(np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)

    # group_sizes по impr_key (сорировка уже сделана выше)
    keys = df["impr_key"].values
    et = df["entity_type"].astype(str).values
    group_sizes: list[int] = []
    group_keys: list[int] = []
    et_per_group: list[str] = []
    cur_key = None
    cur_n = 0
    cur_et = ""
    for k, e in zip(keys, et):
        if k != cur_key:
            if cur_key is not None:
                group_sizes.append(cur_n)
                group_keys.append(int(cur_key))
                et_per_group.append(cur_et)
            cur_key = k
            cur_et = e
            cur_n = 0
        cur_n += 1
    if cur_key is not None:
        group_sizes.append(cur_n)
        group_keys.append(int(cur_key))
        et_per_group.append(cur_et)

    return X, y, group_sizes, w, group_keys, pd.Series(et_per_group, index=group_keys)


def build_synthetic_coview_groups(
    art: Artifacts,
    neighbors_idx: dict[int, np.ndarray],
    neighbors_score: dict[int, np.ndarray],
    *,
    n_groups: int = 100_000,
    group_size: int = 6,
    weight: float = 0.3,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, list[int], pd.Series]:
    """Синтетические pseudo-impression-группы из coview top-K.

    Для каждой выбранной статьи-источника создаётся группа из `group_size`
    кандидатов: топ-1 по coview помечается y=1, остальные ниже по списку — y=0.
    Это даёт LambdaRank понимание «coview top-1 лучше, чем coview lower»,
    не подменяя реальные клики (вес = 0.3, чтобы не доминировать).
    """

    rng = np.random.default_rng(seed)
    cache = prepare_feature_cache(art)

    eligible = [s for s, neigh in neighbors_idx.items() if len(neigh) >= group_size + 1]
    if not eligible:
        return (
            np.zeros((0, len(LTR_FEATURE_NAMES)), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
            np.zeros(0, dtype=np.float32),
            [],
            pd.Series(dtype=str),
        )

    if len(eligible) > n_groups:
        eligible = rng.choice(eligible, size=n_groups, replace=False).tolist()

    rows = []
    for src_idx in eligible:
        neigh = neighbors_idx[src_idx]
        scores = neighbors_score[src_idx]
        n = len(neigh)
        # позитив = top-1, негативы = (group_size-1) случайных из 5..n
        pos_cand = int(neigh[0])
        pos_score = float(scores[0])
        pos_rank = 0
        lower_pool = list(range(min(5, n - 1), n))
        if len(lower_pool) < group_size - 1:
            lower_pool = list(range(1, n))
        neg_choice = rng.choice(lower_pool, size=group_size - 1, replace=False)
        rows.append((int(src_idx), pos_cand, pos_score, pos_rank, 1))
        for r in neg_choice:
            rows.append((int(src_idx), int(neigh[int(r)]), float(scores[int(r)]), int(r), 0))

    df_synth = pd.DataFrame(rows, columns=["src_idx", "tgt_idx", "coview_score", "coview_rank", "y"])

    src_idx_arr = df_synth["src_idx"].values.astype(np.int64)
    cand_idx_arr = df_synth["tgt_idx"].values.astype(np.int64)
    y = df_synth["y"].astype(np.float32).values

    src_views = cache["log_views"][src_idx_arr]
    src_like = cache["like_rate"][src_idx_arr]
    src_fresh = cache["fresh"][src_idx_arr]
    src_age = cache["age_days"][src_idx_arr]
    src_author = cache["author_code"][src_idx_arr]
    src_dept = cache["dept_code"][src_idx_arr]
    src_rub = cache["rub_code"][src_idx_arr]

    cand_views = cache["log_views"][cand_idx_arr]
    cand_log_comments = cache["log_comments"][cand_idx_arr]
    cand_like = cache["like_rate"][cand_idx_arr]
    cand_comment = cache["comment_rate"][cand_idx_arr]
    cand_fav = cache["fav_rate"][cand_idx_arr]
    cand_fresh = cache["fresh"][cand_idx_arr]
    cand_age = cache["age_days"][cand_idx_arr]
    cand_author = cache["author_code"][cand_idx_arr]
    cand_dept = cache["dept_code"][cand_idx_arr]
    cand_rub = cache["rub_code"][cand_idx_arr]

    same_author = (cand_author == src_author).astype(np.float32)
    abs_age_diff = np.abs(cand_age - src_age).astype(np.float32)
    same_dept = (cand_dept == src_dept).astype(np.float32)
    same_rub = (cand_rub == src_rub).astype(np.float32)

    # sim по эмбеддингам — небольшие группы, считаем сразу
    sim = np.einsum("ij,ij->i", art.X[src_idx_arr], art.X[cand_idx_arr], optimize=True).astype(np.float32)
    pos_f = np.zeros(len(df_synth), dtype=np.float32)  # «позиция» неопределена → 0

    X = np.column_stack(
        [
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
            src_views,
            src_like,
            src_fresh,
            pos_f,
            df_synth["coview_score"].astype(np.float32).values,
            df_synth["coview_rank"].astype(np.float32).values,
        ]
    ).astype(np.float32)

    n_groups_built = len(eligible)
    group_sizes = [group_size] * n_groups_built
    group_keys = [-(i + 1) for i in range(n_groups_built)]
    et_per_group = ["__synthetic_coview__"] * n_groups_built
    w = np.full(len(df_synth), weight, dtype=np.float32)

    return X, y, group_sizes, w, group_keys, pd.Series(et_per_group, index=group_keys)


# ---------------------------------------------------------------------------
# trainers
# ---------------------------------------------------------------------------


def train_lgbm(X, y, group, *, sample_weight=None, seed: int = RANDOM_SEED):
    import lightgbm as lgb

    df = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(df, y, group=group, sample_weight=sample_weight)
    return model


def train_catboost(X, y, group, *, sample_weight=None, seed: int = RANDOM_SEED):
    """CatBoostRanker с YetiRank. Pairwise-лоссы CatBoost не поддерживают
    sample_weight, поэтому вес здесь не передаём — зато пишем тот же
    feature_names и используем groupwise YetiRank.
    """

    from catboost import CatBoostRanker, Pool

    group_id = np.repeat(np.arange(len(group), dtype=np.int64), np.asarray(group, dtype=np.int64))
    pool = Pool(data=X, label=y, group_id=group_id, feature_names=LTR_FEATURE_NAMES)
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        random_seed=seed,
        verbose=0,
    )
    model.fit(pool)
    return model


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict_scores(model, X: np.ndarray, name: str) -> np.ndarray:
    if name.startswith("cat"):
        return np.asarray(model.predict(X), dtype=np.float32)
    df = pd.DataFrame(X, columns=LTR_FEATURE_NAMES)
    return np.asarray(model.predict(df), dtype=np.float32)


def eval_metrics(
    y: np.ndarray,
    score: np.ndarray,
    group: list[int],
    *,
    ks: tuple[int, ...] = EVAL_KS,
    min_group_size: int = 2,
) -> dict[str, float]:
    out: dict[str, list[float]] = {f"ndcg@{k}": [] for k in ks}
    out.update({f"mrr@{k}": [] for k in ks})

    pos = 0
    skipped_small = 0
    skipped_no_pos = 0
    for gsz in group:
        sl = slice(pos, pos + gsz)
        yi = y[sl]
        si = score[sl]
        pos += gsz
        if gsz < min_group_size:
            skipped_small += 1
            continue
        if yi.sum() == 0:
            skipped_no_pos += 1
            continue
        order = np.argsort(-si, kind="stable")
        ranked_y = yi[order]
        for k in ks:
            ranked_k = ranked_y[:k]
            out[f"ndcg@{k}"].append(float(ndcg_score([yi], [si], k=k)))
            hits = np.where(ranked_k > 0)[0]
            if len(hits):
                out[f"mrr@{k}"].append(1.0 / (hits[0] + 1))
            else:
                out[f"mrr@{k}"].append(0.0)
    total_eval = len(out["mrr@3"])
    print(
        f"[eval] groups eval={total_eval:,}  small<{min_group_size}={skipped_small:,}  no_pos={skipped_no_pos:,}"
    )
    return {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}


def eval_per_entity(
    y: np.ndarray,
    score: np.ndarray,
    group: list[int],
    et_per_group: pd.Series,
    *,
    k: int = 6,
    min_group_size: int = 2,
) -> pd.DataFrame:
    rows = []
    pos = 0
    by_et: dict[str, list[tuple[float, float]]] = {}
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
    for et, vals in by_et.items():
        rows.append({
            "entity_type": et,
            "n_groups": len(vals),
            f"ndcg@{k}": float(np.mean([v[0] for v in vals])),
            f"mrr@{k}": float(np.mean([v[1] for v in vals])),
        })
    return pd.DataFrame(rows).sort_values("n_groups", ascending=False)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot_propensity(propensity: dict[int, float], out: Path) -> None:
    items = sorted(propensity.items())
    xs, ys = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(x) for x in xs], ys, color="#3478c6")
    for x, y in zip(xs, ys):
        ax.text(str(x), y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Propensity p̂(click | position) на train")
    ax.set_xlabel("article_position")
    ax.set_ylabel("p̂")
    ax.set_ylim(0, max(ys) * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_overall_metrics(metrics: dict, out: Path) -> None:
    """metrics like:
    {'lgbm': {'ndcg@3':..,'ndcg@6':..,'ndcg@10':..,'mrr@..'},
     'catboost': {...}, 'baseline_sim': {...}}"""

    models = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    palette = ["#3478c6", "#f49a3b", "#7d8c8c", "#84c684"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(models):
        ys = [metrics[name][m] for m in metric_names]
        bars = ax.bar(x + i * width - 0.4 + width / 2, ys, width=width, label=name, color=palette[i % len(palette)])
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15)
    ax.set_title("LTR vs baseline (semantic-only) на test")
    ax.set_ylim(0, max(max(metrics[m].values()) for m in models) * 1.25)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_feature_importance(model, names: list[str], out: Path, *, kind: str = "gain") -> None:
    booster = model.booster_
    if kind == "gain":
        imp = booster.feature_importance(importance_type="gain")
    else:
        imp = booster.feature_importance(importance_type="split")
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order][::-1], [imp[i] for i in order][::-1], color="#3478c6")
    for y, val in enumerate([imp[i] for i in order][::-1]):
        ax.text(val, y, f" {val:,.0f}", va="center", fontsize=8)
    ax.set_title(f"Feature importance LightGBM ({kind})")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_entity(df_metrics: pd.DataFrame, out: Path, k: int = 6) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    df_metrics = df_metrics.sort_values("n_groups", ascending=False).head(8)
    et_short = [re.sub(r"^article\.", "", x) for x in df_metrics["entity_type"]]
    et_short = [(s[:36] + "…") if len(s) > 36 else s for s in et_short]
    width = 0.4
    x = np.arange(len(df_metrics))
    ax.bar(x - width / 2, df_metrics[f"ndcg@{k}"], width=width, label=f"nDCG@{k}", color="#3478c6")
    ax.bar(x + width / 2, df_metrics[f"mrr@{k}"], width=width, label=f"MRR@{k}", color="#f49a3b")
    ax.set_xticks(x)
    ax.set_xticklabels(et_short, rotation=22, ha="right", fontsize=9)
    ax.set_title(f"LTR метрики по entity_type (test, LightGBM, k={k})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(df_metrics["n_groups"]):
        ax.text(i, max(df_metrics[f"ndcg@{k}"].max(), df_metrics[f"mrr@{k}"].max()) * 1.05,
                f"n={n:,}", ha="center", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_group_sizes(group_sizes: list[int], out: Path) -> None:
    s = pd.Series(group_sizes).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(s.index.astype(str), s.values, color="#3478c6")
    for x, y in zip(s.index, s.values):
        ax.text(str(x), y, f"{y:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Распределение размеров impression-групп (train)")
    ax.set_xlabel("размер группы (число кандидатов)")
    ax.set_ylabel("число групп")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_target_share_by_position(logs: pd.DataFrame, out: Path) -> None:
    g = logs.groupby("article_position", observed=True)["target"].agg(["mean", "size"])
    g = g[g.index.to_series().between(0, 12)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g.index.astype(str), g["mean"], color="#3478c6")
    for x, y, n in zip(g.index, g["mean"], g["size"]):
        ax.text(str(x), y, f"{y:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set_title("Доля кликов по позициям (raw, без коррекции)")
    ax.set_xlabel("article_position")
    ax.set_ylabel("share clicked")
    ax.set_ylim(0, g["mean"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_train_test_split(by_day: pd.Series, cut: pd.Timestamp, out: Path) -> None:
    cut_naive = cut.tz_convert(None) if cut.tz is not None else cut
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#3478c6" if pd.Timestamp(d) < cut_naive else "#f49a3b" for d in by_day.index]
    ax.bar([str(d) for d in by_day.index], by_day.values, color=colors)
    ax.set_xticks(np.arange(0, len(by_day), max(1, len(by_day) // 10)))
    ax.set_xticklabels([str(by_day.index[i]) for i in np.arange(0, len(by_day), max(1, len(by_day) // 10))], rotation=30, ha="right")
    ax.set_title(f"Train/test time-split (cut = {cut.date()} UTC)")
    ax.set_ylabel("число строк логов")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_score_distribution(score_clicked: np.ndarray, score_unclicked: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(min(score_unclicked.min(), score_clicked.min()),
                       max(score_unclicked.max(), score_clicked.max()), 60)
    ax.hist(score_unclicked, bins=bins, alpha=0.55, label="not clicked", color="#7d8c8c", density=True)
    ax.hist(score_clicked, bins=bins, alpha=0.65, label="clicked", color="#f49a3b", density=True)
    ax.set_title("Распределение LTR-скора: clicked vs not clicked (test)")
    ax.set_xlabel("score")
    ax.set_ylabel("плотность")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[1/8] loading articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)
    print(f"      aligned articles: {art.df.shape[0]:,}")

    print("[1b/8] loading coview index…")
    coview_neigh_idx, coview_neigh_score, coview_pair_df = load_coview_aligned(art)

    print("[2/8] loading logs…")
    logs = load_logs()
    print(f"      mem dataframe: ~{logs.memory_usage(deep=True).sum()/1e6:.0f} MB")

    train_logs, test_logs, cut = time_split(logs, train_quantile=0.8)
    daily_counts = logs.groupby(logs["hit_dttm"].dt.date).size()
    del logs

    print("[3/8] propensity (train)…")
    propensity = estimate_position_propensity(train_logs)
    print("      propensity:", {k: round(v, 4) for k, v in sorted(propensity.items())})

    print("[3b/8] saving early plots…")
    plot_target_share_by_position(train_logs, FIGURES_DIR / "target_share_by_position.png")
    plot_train_test_split(daily_counts, cut, FIGURES_DIR / "train_test_split_dates.png")
    plot_propensity(propensity, FIGURES_DIR / "propensity_by_position.png")

    print("[4/8] building TRAIN dataset…")
    Xtr, ytr, gtr, wtr, _, _ = build_dataset(
        art, train_logs, propensity=propensity, max_groups=600_000,
        coview_pair_df=coview_pair_df,
    )
    del train_logs
    print(f"      train rows: {len(ytr):,}  groups: {len(gtr):,}  positives: {int(ytr.sum()):,}")
    plot_group_sizes(gtr, FIGURES_DIR / "group_size_distribution.png")

    print("[4b/8] building synthetic coview groups…")
    Xs, ys, gs, ws, _, _ = build_synthetic_coview_groups(
        art, coview_neigh_idx, coview_neigh_score, n_groups=80_000
    )
    if len(ys):
        Xtr = np.vstack([Xtr, Xs])
        ytr = np.concatenate([ytr, ys])
        gtr = list(gtr) + list(gs)
        wtr = np.concatenate([wtr, ws])
        print(f"      added synth: rows={len(ys):,}  groups={len(gs):,}  positives={int(ys.sum()):,}")
        print(f"      total train: rows={len(ytr):,}  groups={len(gtr):,}  positives={int(ytr.sum()):,}")

    print("[5/8] training LightGBM…")
    t0 = time.time()
    model_lgbm = train_lgbm(Xtr, ytr, gtr, sample_weight=wtr)
    print(f"      lgbm trained in {time.time()-t0:.1f}s")

    print("[5b/8] training CatBoost (опционально, медленнее)…")
    t0 = time.time()
    try:
        model_cb = train_catboost(Xtr, ytr, gtr, sample_weight=wtr)
        print(f"      catboost trained in {time.time()-t0:.1f}s")
    except Exception as e:
        print("      catboost skipped:", e)
        model_cb = None

    print("[6/8] building TEST dataset…")
    Xte, yte, gte, wte, keys_te, et_te = build_dataset(
        art, test_logs, max_groups=150_000, coview_pair_df=coview_pair_df,
    )
    del test_logs
    print(f"      test rows: {len(yte):,}  groups: {len(gte):,}  positives: {int(yte.sum()):,}")

    print("[7/8] evaluating…")
    s_lgbm = predict_scores(model_lgbm, Xte, "lgbm")
    s_cb = predict_scores(model_cb, Xte, "cb") if model_cb is not None else None
    s_baseline_sim = Xte[:, 0]  # фича sim
    rng = np.random.default_rng(123)
    s_random = rng.standard_normal(len(yte)).astype(np.float32)

    metrics = {
        "lgbm": eval_metrics(yte, s_lgbm, gte),
        "baseline_sim": eval_metrics(yte, s_baseline_sim, gte),
        "random": eval_metrics(yte, s_random, gte),
    }
    if s_cb is not None:
        metrics["catboost"] = eval_metrics(yte, s_cb, gte)

    per_entity_lgbm = eval_per_entity(yte, s_lgbm, gte, et_te, k=6)
    per_entity_baseline = eval_per_entity(yte, s_baseline_sim, gte, et_te, k=6)
    per_entity = per_entity_lgbm.merge(
        per_entity_baseline.rename(columns={"ndcg@6": "ndcg@6_baseline", "mrr@6": "mrr@6_baseline"}).drop(columns=["n_groups"]),
        on="entity_type", how="left",
    )

    print("[8/8] saving artifacts…")
    with open(MODELS_DIR / "ltr_lgbm.pkl", "wb") as f:
        pickle.dump({"model": model_lgbm, "feature_names": LTR_FEATURE_NAMES, "propensity": propensity}, f)
    if model_cb is not None:
        model_cb.save_model(str(MODELS_DIR / "ltr_catboost.cbm"))

    full_metrics = {
        "cut": str(cut),
        "train_rows": int(len(ytr)),
        "train_groups": int(len(gtr)),
        "train_positives": int(ytr.sum()),
        "test_rows": int(len(yte)),
        "test_groups": int(len(gte)),
        "test_positives": int(yte.sum()),
        "propensity": {int(k): float(v) for k, v in propensity.items()},
        "overall": metrics,
        "per_entity_top": per_entity.head(8).to_dict(orient="records"),
    }
    (REPORTS_DIR / "ltr_full_metrics.json").write_text(
        json.dumps(full_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    per_entity.to_csv(REPORTS_DIR / "ltr_full_per_entity.csv", index=False)

    plot_overall_metrics(metrics, FIGURES_DIR / "ltr_metrics_overall.png")
    plot_feature_importance(model_lgbm, LTR_FEATURE_NAMES, FIGURES_DIR / "feature_importance_gain.png", kind="gain")
    plot_per_entity(per_entity_lgbm, FIGURES_DIR / "ltr_metrics_by_entity.png", k=6)
    plot_score_distribution(
        s_lgbm[yte > 0], s_lgbm[yte == 0], FIGURES_DIR / "score_distribution_clicked_vs_not.png"
    )

    print("\n=== DONE ===")
    print("model:", MODELS_DIR / "ltr_lgbm.pkl")
    if model_cb is not None:
        print("model:", MODELS_DIR / "ltr_catboost.cbm")
    print("metrics:", REPORTS_DIR / "ltr_full_metrics.json")
    print("figures:", sorted(FIGURES_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
