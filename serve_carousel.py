"""Единый inference-скрипт сборки рекомендательной карусели i2i для Т‑Ж.

Реализует ровно ту архитектуру, что описана в дипломе и нарисована на
`recs_system_diagram.png`:

  source article a
        |
        +--> similar-пул: kNN top-K_sim (по cosine в эмбеддингах)
        |                 ∪ coview top-M (поведенческий i2i)
        +--> explore-пул: per-dept top-N + cross-dept top-N
                          из quality/trend пулов
        |
        v
   LTR (LightGBM LambdaRank) — скоринг 17 фичей (включая coview_score,
                                coview_rank)
        |
        v
   Reranking + продуктовые правила:
     - дедуп по нормализованному заголовку
     - лимиты по автору и рубрике
     - cross-department blacklist
     - фильтр explore по сходству и возрасту
     - интерливинг 9 similar + 3 explore
        |
        v
   Карусель K=12

Использует уже подготовленные артефакты:
  - embeddings_cache/        (memmap-эмбеддинги)
  - coview_cache/coview_index.pkl.gz
  - models/ltr_lgbm.pkl      (сохранённая LightGBM-модель)

CLI:
  python serve_carousel.py --source <article_id>          # одна карусель в stdout
  python serve_carousel.py --sources-csv path.csv --out path.csv  # пакетно
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_ltr_full import (
    COVIEW_PATH,
    COVIEW_RANK_MISSING,
    COVIEW_SCORE_MISSING,
    LTR_FEATURE_NAMES,
    MODELS_DIR,
    Artifacts,
    fit_retriever,
    load_articles,
    load_coview_aligned,
    load_embeddings_memmap,
    prepare_feature_cache,
    safe_str,
)

# ---------------------------------------------------------------------------
# constants — настройки пулов, ровно те, что описаны в дипломе/диаграмме
# ---------------------------------------------------------------------------

K_CAROUSEL = 12
EXPLORE_SLOTS = 3
TOPK_SIM = 100  # kNN top-100 ближайших по cosine
COVIEW_TOP_M = 50  # сколько coview-соседей подмешать в similar
EXPLORE_POOL_TOPM = 3000  # per-dept и global top-M по quality+trend
EXPLORE_POOL_SAME = 30  # per-source: сколько брать из same-dept
EXPLORE_POOL_CROSS = 20  # per-source: сколько брать из cross-dept
MAX_SAME_AUTHOR = 2
MAX_SAME_RUBRIC = 6
MAX_CANDIDATE_AGE_DAYS = 365.0
EXPLORE_SIM_THR_SAME = 0.55
EXPLORE_SIM_THR_CROSS = 0.45
EXPLORE_FIRST_POS = 4
EXPLORE_STEP = 3

DEPT_CROSS_BLACKLIST: dict[str, set[str]] = {
    "Медицина": {"Еда"},
}

# ---- эвристический fallback‑скор ------------------------------------------
# Используется в двух случаях:
#   1) глобально — если models/ltr_lgbm.pkl не удалось загрузить;
#   2) per‑source — если у конкретного источника не хватает данных для LTR
#      (нет coview‑соседей И статья опубликована недавно).
# Формула:
#   score = sim
#         + HEURISTIC_WEIGHT_VIEWS  * log(1 + views_candidate)
#         - HEURISTIC_WEIGHT_AGE    * min(age, 365) / 365
#         + HEURISTIC_WEIGHT_RUBRIC * 1[rubric_candidate == rubric_source]
# Веса подобраны так, чтобы вклад priors был сопоставим с диапазоном sim ∈ [0, 1],
# но не доминировал над похожестью.
HEURISTIC_WEIGHT_VIEWS = 0.05
HEURISTIC_WEIGHT_AGE = 0.15
HEURISTIC_WEIGHT_RUBRIC = 0.10

# Per‑source fallback срабатывает, если ОБА условия выполнены:
#   - в coview‑индексе у источника меньше MIN_COVIEW_NEIGHBORS соседей
#   - И источник опубликован менее MIN_SOURCE_AGE_DAYS дней назад
# Логика: если только одно из условий — LTR ещё имеет какой‑то сигнал.
# Если оба — coview‑фичи будут заполнителями, src‑фичи (views/like_rate) не
# успели накопиться, скоры LTR станут малоинформативными.
MIN_COVIEW_NEIGHBORS = 3
MIN_SOURCE_AGE_DAYS = 3.0

_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RE_NUM = re.compile(r"\b\d+([.,]\d+)?\b")
_RE_SPACE = re.compile(r"\s+")


def normalize_title_for_dedup(title: str) -> str:
    t = safe_str(title).lower().replace("ё", "е")
    t = _RE_YEAR.sub(" <year> ", t)
    t = _RE_NUM.sub(" <num> ", t)
    t = re.sub(r"[\"'“”«»()\[\]{}:;,.!?/\\|—–-]+", " ", t)
    t = _RE_SPACE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------


def _row_quality(views: float, like_rate: float, comment_rate: float, fav_rate: float) -> float:
    z = 12.0 * (0.7 * like_rate + 0.3 * fav_rate) + 4.0 * comment_rate
    return float(1.0 / (1.0 + math.exp(-z)))


def _row_trend(views: float, days: float) -> float:
    return float(math.log1p(max(views, 0.0)) / math.sqrt(max(days, 0.0) + 1.0))


def build_explore_pools(
    art: Artifacts, *, top_m: int = EXPLORE_POOL_TOPM
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Готовит per-department top-M и глобальный top-M по 0.6·quality + 0.4·trend.

    Считается один раз для всего каталога; используется потом для всех источников.
    """

    df = art.df.copy()
    views = pd.to_numeric(df.get("article_stats__stats_views", 0.0), errors="coerce").fillna(0.0).values
    lr = pd.to_numeric(df.get("article_stats__like_rate", 0.0), errors="coerce").fillna(0.0).values
    cr = pd.to_numeric(df.get("article_stats__comment_rate", 0.0), errors="coerce").fillna(0.0).values
    favs = pd.to_numeric(df.get("article_stats__stats_favorites", 0.0), errors="coerce").fillna(0.0).values
    days = pd.to_numeric(df.get("article_dates__days_since_published", 365.0), errors="coerce").fillna(365.0).values

    fav_rate = favs / np.maximum(views, 1.0)
    quality = 1.0 / (1.0 + np.exp(-(12.0 * (0.7 * lr + 0.3 * fav_rate) + 4.0 * cr)))
    trend = np.log1p(np.clip(views, 0, None)) / np.sqrt(days + 1.0)
    df["_quality"] = quality
    df["_trend"] = trend
    df["_explore_score"] = 0.6 * quality + 0.4 * trend

    per_dept: dict[str, pd.DataFrame] = {}
    if "article_base__department" in df.columns:
        for dept, g in df.groupby("article_base__department", dropna=False):
            d = safe_str(dept)
            if not d:
                continue
            per_dept[d] = g.sort_values("_explore_score", ascending=False).head(top_m).copy()

    global_pool = df.sort_values("_explore_score", ascending=False).head(top_m).copy()
    return per_dept, global_pool


def get_similar_candidates(
    art: Artifacts,
    src_idx: int,
    *,
    topk_sim: int = TOPK_SIM,
    coview_neigh_idx: dict[int, np.ndarray] | None = None,
    coview_neigh_score: dict[int, np.ndarray] | None = None,
    coview_top_m: int = COVIEW_TOP_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Similar-пул: kNN top-K по cosine ∪ coview top-M.

    Возвращает (cand_idx, sim_to_source). Если кандидат пришёл и из kNN,
    и из coview — берётся max(sim_emb, sigmoid(coview_score) * 1.0).
    """

    dist, idx = art.nn.kneighbors(art.X[src_idx : src_idx + 1], n_neighbors=topk_sim + 1)
    idx = idx[0]
    sim = (1.0 - dist[0]).astype(np.float32)
    mask = idx != src_idx
    idx = idx[mask].astype(np.int64)
    sim = sim[mask]

    if coview_neigh_idx is not None and src_idx in coview_neigh_idx:
        cv_idx = coview_neigh_idx[src_idx][:coview_top_m]
        cv_score = coview_neigh_score[src_idx][:coview_top_m]
        already = {int(j): k for k, j in enumerate(idx.tolist())}
        add_idx: list[int] = []
        add_sim: list[float] = []
        for j, sc in zip(cv_idx.tolist(), cv_score.tolist()):
            if int(j) == src_idx:
                continue
            cov_sigmoid = 1.0 / (1.0 + math.exp(-float(sc)))
            if int(j) in already:
                sim[already[int(j)]] = max(float(sim[already[int(j)]]), cov_sigmoid)
            else:
                add_idx.append(int(j))
                add_sim.append(cov_sigmoid)
        if add_idx:
            idx = np.concatenate([idx, np.asarray(add_idx, dtype=np.int64)])
            sim = np.concatenate([sim, np.asarray(add_sim, dtype=np.float32)])

    return idx, sim


def get_explore_candidates(
    art: Artifacts,
    src_idx: int,
    *,
    per_dept_pool: dict[str, pd.DataFrame],
    global_pool: pd.DataFrame,
    pool_size_same: int = EXPLORE_POOL_SAME,
    pool_size_cross: int = EXPLORE_POOL_CROSS,
) -> pd.DataFrame:
    """Explore-пул для одного источника: top-N same-dept + top-N cross-dept."""

    src = art.df.iloc[src_idx]
    src_dept = safe_str(src.get("article_base__department", ""))
    src_id = safe_str(src.get("article_id", ""))

    same = per_dept_pool.get(src_dept)
    if same is None:
        same = global_pool
    same = same.head(pool_size_same).copy()

    cross = global_pool.head(pool_size_cross * 4).copy()  # с запасом, отфильтруем ниже
    if src_dept and "article_base__department" in cross.columns:
        cross = cross[cross["article_base__department"].map(safe_str) != src_dept]
    forbidden = DEPT_CROSS_BLACKLIST.get(src_dept, set())
    if forbidden and "article_base__department" in cross.columns:
        cross = cross[~cross["article_base__department"].map(safe_str).isin(forbidden)]
    cross = cross.head(pool_size_cross).copy()

    out = pd.concat([same, cross], ignore_index=True)
    if src_id and "article_id" in out.columns:
        out = out[out["article_id"].map(safe_str) != src_id]
    out = out.drop_duplicates(subset=["article_id"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# LTR scoring (17 features, identical to train_ltr_full)
# ---------------------------------------------------------------------------


def build_features(
    art: Artifacts,
    cache: dict[str, np.ndarray],
    src_idx: int,
    cand_idx: np.ndarray,
    sim: np.ndarray,
    coview_neigh_idx: dict[int, np.ndarray] | None,
    coview_neigh_score: dict[int, np.ndarray] | None,
) -> pd.DataFrame:
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
        for j, c in enumerate(cand_idx.tolist()):
            r = cand_to_pos.get(int(c))
            if r is not None:
                coview_score[j] = float(scores[r])
                coview_rank[j] = float(r)

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
            np.full(n, src_views, dtype=np.float32),
            np.full(n, src_like, dtype=np.float32),
            np.full(n, src_fresh, dtype=np.float32),
            pos_f,
            coview_score,
            coview_rank,
        ]
    ).astype(np.float32)
    return pd.DataFrame(X, columns=LTR_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Heuristic fallback scorer — используется когда LTR недоступен или у источника
# не хватает данных. Формула описана у констант HEURISTIC_WEIGHT_*.
# ---------------------------------------------------------------------------


def heuristic_score(
    sim: np.ndarray,
    cand_log_views: np.ndarray,
    cand_age_days: np.ndarray,
    rubric_match: np.ndarray,
) -> np.ndarray:
    """Эвристический скор «похожесть + priors по качеству/свежести/рубрике».

    Все массивы — длиной N (число кандидатов). Возвращает np.ndarray[float32].
    """

    sim_f = np.asarray(sim, dtype=np.float32)
    views_f = np.asarray(cand_log_views, dtype=np.float32)
    age_norm = (np.clip(np.asarray(cand_age_days, dtype=np.float32), 0.0, 365.0) / 365.0).astype(np.float32)
    rub_f = np.asarray(rubric_match, dtype=np.float32)
    return (
        sim_f
        + HEURISTIC_WEIGHT_VIEWS * views_f
        - HEURISTIC_WEIGHT_AGE * age_norm
        + HEURISTIC_WEIGHT_RUBRIC * rub_f
    ).astype(np.float32)


def _score_candidates(
    cache: dict[str, np.ndarray],
    src_idx: int,
    cand_idx: np.ndarray,
    sim_vals: np.ndarray,
) -> np.ndarray:
    """Вспомогательная обёртка над heuristic_score, выдёргивает нужные поля
    из feature‑cache для набора кандидатов и считает скор.
    """

    src_rub = int(cache["rub_code"][src_idx])
    cand_rub = cache["rub_code"][cand_idx]
    rubric_match = (cand_rub == src_rub).astype(np.float32)
    return heuristic_score(
        sim_vals,
        cache["log_views"][cand_idx],
        cache["age_days"][cand_idx],
        rubric_match,
    )


def _is_sparse_source(
    art: Artifacts,
    src_idx: int,
    coview_neigh_idx: dict[int, np.ndarray] | None,
    *,
    min_coview: int = MIN_COVIEW_NEIGHBORS,
    min_age_days: float = MIN_SOURCE_AGE_DAYS,
) -> bool:
    """True, если у источника не хватает данных для уверенного LTR‑скоринга:
    в coview‑индексе меньше min_coview соседей И статья опубликована меньше
    min_age_days дней назад. Только при ОБОИХ условиях — иначе LTR ещё имеет
    полезный сигнал из одной из частей фичей.
    """

    src_age = float(art.df.iloc[src_idx].get("article_dates__days_since_published", 365.0))
    n_coview = 0
    if coview_neigh_idx is not None and src_idx in coview_neigh_idx:
        n_coview = int(len(coview_neigh_idx[src_idx]))
    return n_coview < min_coview and src_age < min_age_days


# ---------------------------------------------------------------------------
# Reranking — продуктовые правила и сборка карусели K=12
# ---------------------------------------------------------------------------


def _max_cosine_to(art: Artifacts, picked_idx: list[int], cand_idx: int) -> float:
    if not picked_idx:
        return 0.0
    v = art.X[cand_idx]
    M = art.X[np.asarray(picked_idx, dtype=np.int64)]
    return float(np.max(M @ v))


def rerank_carousel(
    art: Artifacts,
    src_idx: int,
    sim_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    *,
    k: int = K_CAROUSEL,
    explore_slots: int = EXPLORE_SLOTS,
    max_same_author: int = MAX_SAME_AUTHOR,
    max_same_rubric: int = MAX_SAME_RUBRIC,
    max_candidate_age_days: float | None = MAX_CANDIDATE_AGE_DAYS,
    explore_first_pos: int = EXPLORE_FIRST_POS,
    explore_step: int = EXPLORE_STEP,
) -> list[dict]:
    """Применяет продуктовые правила и собирает финальную карусель K=12."""

    src = art.df.iloc[src_idx]
    src_id = safe_str(src.get("article_id", ""))
    src_dept = safe_str(src.get("article_base__department", ""))
    src_rubric = safe_str(src.get("article_base__rubric", ""))
    src_title_norm = normalize_title_for_dedup(src.get("article_base__title", ""))

    target_sim = max(0, k - explore_slots)

    # --- фильтр возраста (по умолчанию 365 дней)
    def _age_ok(row: pd.Series) -> bool:
        if max_candidate_age_days is None:
            return True
        a = row.get("article_dates__days_since_published")
        if pd.isna(a):
            return True
        return float(a) <= float(max_candidate_age_days)

    # --- similar: greedy с диверсификацией ----------------------------------
    used_title_norm: set[str] = {src_title_norm} if src_title_norm else set()
    author_cnt: dict[str, int] = {}
    rubric_cnt: dict[str, int] = {}
    picked_idx: list[int] = []
    similar_items: list[dict] = []

    sim_sorted = sim_df.sort_values("ltr_score", ascending=False)
    for _, r in sim_sorted.iterrows():
        if len(similar_items) >= target_sim:
            break
        if not _age_ok(r):
            continue
        cid = safe_str(r.get("article_id", ""))
        if not cid or cid == src_id:
            continue
        title = safe_str(r.get("article_base__title", ""))
        tnorm = normalize_title_for_dedup(title)
        if tnorm and tnorm in used_title_norm:
            continue
        aid = safe_str(r.get("article_base__author_id", ""))
        rub = safe_str(r.get("article_base__rubric", ""))
        if aid and author_cnt.get(aid, 0) >= max_same_author:
            continue
        if rub and rubric_cnt.get(rub, 0) >= max_same_rubric:
            continue
        ridx = int(r["row_idx"])
        similar_items.append(
            {
                "source_article_id": src_id,
                "candidate_article_id": cid,
                "mix": "similar",
                "score": float(r["ltr_score"]),
                "similarity": float(r["sim"]),
                "candidate_title": title,
                "candidate_department": safe_str(r.get("article_base__department", "")),
                "candidate_rubric": rub,
            }
        )
        picked_idx.append(ridx)
        if tnorm:
            used_title_norm.add(tnorm)
        if aid:
            author_cnt[aid] = author_cnt.get(aid, 0) + 1
        if rub:
            rubric_cnt[rub] = rubric_cnt.get(rub, 0) + 1

    # --- explore: квоты same/cross + anti-sim к уже отобранным ---------------
    explore_same_target = max(0, explore_slots - 1) if src_dept else explore_slots
    explore_cross_target = 1 if src_dept and explore_slots > 0 else 0
    explore_same_added = 0
    explore_cross_added = 0
    explore_items: list[dict] = []

    # Детерминированная shuffle-по-источнику, чтобы две статьи с похожим source не давали одинаковые explore
    seed = int.from_bytes(hashlib.blake2b(src_id.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)

    exp_sorted = exp_df.sort_values("ltr_score", ascending=False).reset_index(drop=True)
    # softmax-сэмплирование top-30 чтобы разнообразить explore между источниками
    pool = exp_sorted.head(30).copy()
    if len(pool) > 0:
        w = pool["ltr_score"].astype(float).values
        z = (w - float(np.max(w))) / 0.35
        p = np.exp(np.clip(z, -30, 30))
        p = p / max(float(p.sum()), 1e-12)
        order = rng.choice(len(pool), size=len(pool), replace=False, p=p)
        pool = pool.iloc[order].reset_index(drop=True)

    picked_ids = {it["candidate_article_id"] for it in similar_items}

    for _, r in pool.iterrows():
        if len(explore_items) >= explore_slots:
            break
        if not _age_ok(r):
            continue
        cid = safe_str(r.get("article_id", ""))
        if not cid or cid == src_id or cid in picked_ids:
            continue
        title = safe_str(r.get("article_base__title", ""))
        tnorm = normalize_title_for_dedup(title)
        if tnorm and tnorm in used_title_norm:
            continue
        cand_dept = safe_str(r.get("article_base__department", ""))
        cand_rub = safe_str(r.get("article_base__rubric", ""))
        if src_dept:
            if cand_dept == src_dept:
                if explore_same_added >= explore_same_target:
                    continue
                if src_rubric and cand_rub and cand_rub == src_rubric:
                    continue
                sim_thr = EXPLORE_SIM_THR_SAME
            else:
                if explore_cross_added >= explore_cross_target:
                    continue
                forbidden = DEPT_CROSS_BLACKLIST.get(src_dept, set())
                if cand_dept and cand_dept in forbidden:
                    continue
                sim_thr = EXPLORE_SIM_THR_CROSS
        else:
            sim_thr = EXPLORE_SIM_THR_SAME

        ridx = int(r["row_idx"])
        if _max_cosine_to(art, picked_idx, ridx) > sim_thr:
            continue

        explore_items.append(
            {
                "source_article_id": src_id,
                "candidate_article_id": cid,
                "mix": "explore",
                "score": float(r["ltr_score"]),
                "similarity": float(r["sim"]),
                "candidate_title": title,
                "candidate_department": cand_dept,
                "candidate_rubric": cand_rub,
            }
        )
        picked_idx.append(ridx)
        picked_ids.add(cid)
        if tnorm:
            used_title_norm.add(tnorm)
        if src_dept and cand_dept == src_dept:
            explore_same_added += 1
        elif src_dept and cand_dept != src_dept:
            explore_cross_added += 1

    # --- интерливинг similar + explore: позиции 4, 7, 10 при K=12 -----------
    if explore_slots > 0:
        explore_positions = sorted(
            {
                explore_first_pos + i * explore_step
                for i in range(explore_slots)
                if 1 <= explore_first_pos + i * explore_step <= k
            }
        )
        # внутри карусели explore тоже идут по убыванию ltr_score:
        # softmax-семплинг выше нужен только для разнообразия между источниками,
        # внутри финальной выдачи лучший explore должен стоять на позиции 4.
        explore_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    else:
        explore_positions = []

    merged: list[dict] = []
    si = ei = 0
    for pos in range(1, k + 1):
        want_explore = pos in explore_positions and ei < len(explore_items)
        if want_explore:
            merged.append(explore_items[ei])
            ei += 1
        elif si < len(similar_items):
            merged.append(similar_items[si])
            si += 1
        elif ei < len(explore_items):
            merged.append(explore_items[ei])
            ei += 1
        else:
            break

    for i, row in enumerate(merged, start=1):
        row["rank"] = i
    return merged[:k]


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ServeContext:
    art: Artifacts
    cache: dict[str, np.ndarray]
    model: object | None  # None → используется heuristic_score (см. build_context)
    coview_neigh_idx: dict[int, np.ndarray]
    coview_neigh_score: dict[int, np.ndarray]
    per_dept_pool: dict[str, pd.DataFrame]
    global_pool: pd.DataFrame


def build_context(model_path: Path = MODELS_DIR / "ltr_lgbm.pkl") -> ServeContext:
    """Готовит всё, что нужно для серверного inference (один раз на процесс).

    Если LTR‑модель по `model_path` не загружается (файл отсутствует или
    битый), процесс НЕ падает — `ctx.model` останется `None`, и `serve_one`
    автоматически переключится на эвристический fallback‑скор
    (`heuristic_score`). Это позволяет демонстрировать систему даже без
    обученной модели и переживать перерывы в model registry.
    """

    print("[serve] loading articles + embeddings…")
    df_articles = load_articles()
    X_ids, X_emb = load_embeddings_memmap()
    art = fit_retriever(df_articles, X_ids, X_emb)
    print(f"      articles: {art.df.shape[0]:,}")

    print("[serve] loading LTR model…")
    try:
        bundle = pickle.loads(model_path.read_bytes())
        model = bundle["model"]
        print(f"      loaded: {model_path.name}")
    except (FileNotFoundError, OSError, KeyError, pickle.UnpicklingError) as e:
        print(f"      ⚠ не удалось загрузить LTR‑модель из {model_path}: {e}")
        print("      ⚠ fallback: используем эвристический скор (sim + priors)")
        model = None

    print("[serve] loading coview index…")
    coview_neigh_idx, coview_neigh_score, _ = load_coview_aligned(art)

    print("[serve] preparing feature cache + explore pools…")
    t = time.time()
    cache = prepare_feature_cache(art)
    per_dept, global_pool = build_explore_pools(art)
    print(f"      ready in {time.time()-t:.1f}s")

    return ServeContext(
        art=art,
        cache=cache,
        model=model,
        coview_neigh_idx=coview_neigh_idx,
        coview_neigh_score=coview_neigh_score,
        per_dept_pool=per_dept,
        global_pool=global_pool,
    )


def serve_one(ctx: ServeContext, source_id: str) -> pd.DataFrame:
    """Полный inference для одной статьи-источника. Возвращает карусель K=12."""

    art = ctx.art
    src_idx = art.article_id_to_row.get(source_id)
    if src_idx is None:
        raise KeyError(f"article_id {source_id} not in retriever index")

    # ----- similar pool: kNN ∪ coview
    sim_idx, sim_vals = get_similar_candidates(
        art,
        src_idx,
        topk_sim=TOPK_SIM,
        coview_neigh_idx=ctx.coview_neigh_idx,
        coview_neigh_score=ctx.coview_neigh_score,
        coview_top_m=COVIEW_TOP_M,
    )

    # ----- explore pool: per-dept + cross-dept
    exp_pool = get_explore_candidates(
        art,
        src_idx,
        per_dept_pool=ctx.per_dept_pool,
        global_pool=ctx.global_pool,
    )

    # ----- решаем, использовать LTR или эвристический fallback ---------------
    # Глобальный fallback — если модель не загрузилась (см. build_context).
    # Per‑source fallback — если у источника не хватает данных (нет coview‑
    # соседей и статья опубликована недавно): src‑фичи и coview‑фичи окажутся
    # неинформативными, LTR выродится в шум.
    use_heuristic = (
        ctx.model is None
        or _is_sparse_source(art, src_idx, ctx.coview_neigh_idx)
    )

    # ----- scoring similar
    if use_heuristic:
        sim_score = _score_candidates(ctx.cache, src_idx, sim_idx, sim_vals)
    else:
        sim_feat = build_features(
            art, ctx.cache, src_idx, sim_idx, sim_vals,
            ctx.coview_neigh_idx, ctx.coview_neigh_score,
        )
        sim_score = np.asarray(ctx.model.predict(sim_feat), dtype=np.float32)

    sim_df = art.df.iloc[sim_idx].copy()
    sim_df["row_idx"] = sim_idx
    sim_df["sim"] = sim_vals
    sim_df["ltr_score"] = sim_score

    # ----- scoring explore: считаем sim к источнику + LTR/эвристику
    exp_idx = exp_pool["article_id"].map(art.article_id_to_row.get).values
    exp_mask = np.array([i is not None for i in exp_idx])
    exp_pool = exp_pool[exp_mask].copy().reset_index(drop=True)
    if len(exp_pool) == 0:
        exp_df_out = pd.DataFrame(
            columns=list(art.df.columns) + ["row_idx", "sim", "ltr_score"]
        )
    else:
        exp_idx_arr = np.asarray([art.article_id_to_row[a] for a in exp_pool["article_id"]], dtype=np.int64)
        src_vec = art.X[src_idx]
        exp_sim = (art.X[exp_idx_arr] @ src_vec).astype(np.float32)
        if use_heuristic:
            exp_score = _score_candidates(ctx.cache, src_idx, exp_idx_arr, exp_sim)
        else:
            exp_feat = build_features(
                art, ctx.cache, src_idx, exp_idx_arr, exp_sim,
                ctx.coview_neigh_idx, ctx.coview_neigh_score,
            )
            exp_score = np.asarray(ctx.model.predict(exp_feat), dtype=np.float32)
        # explore слабее по умолчанию (мягкий приоритет similar) — как в ноутбуке
        exp_score = exp_score * 0.85
        exp_df_out = art.df.iloc[exp_idx_arr].copy()
        exp_df_out["row_idx"] = exp_idx_arr
        exp_df_out["sim"] = exp_sim
        exp_df_out["ltr_score"] = exp_score

    # ----- reranking
    rows = rerank_carousel(art, src_idx, sim_df, exp_df_out)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["scoring_mode"] = "heuristic" if use_heuristic else "ltr"
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference карусели i2i по архитектуре из диплома")
    ap.add_argument("--source", help="article_id одной статьи-источника")
    ap.add_argument("--sources-csv", help="CSV/TXT со списком article_id (один на строку)")
    ap.add_argument("--out", help="куда сохранить итог (CSV). Если не задан — печать в stdout")
    ap.add_argument("--model", default=str(MODELS_DIR / "ltr_lgbm.pkl"))
    args = ap.parse_args()

    if not args.source and not args.sources_csv:
        ap.error("укажите --source <article_id> или --sources-csv <path>")

    ctx = build_context(Path(args.model))

    sources: list[str] = []
    if args.source:
        sources.append(args.source)
    if args.sources_csv:
        with open(args.sources_csv, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split(",")[0]
                if s and s.lower() != "article_id":
                    sources.append(s)

    all_rows: list[pd.DataFrame] = []
    for sid in sources:
        try:
            res = serve_one(ctx, sid)
            all_rows.append(res)
        except KeyError as e:
            print(f"  skip {sid}: {e}")

    if not all_rows:
        print("nothing produced.")
        return
    out_df = pd.concat(all_rows, ignore_index=True)

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"saved: {args.out}  ({len(out_df):,} rows)")
    else:
        with pd.option_context("display.max_colwidth", 80, "display.width", 220):
            print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
