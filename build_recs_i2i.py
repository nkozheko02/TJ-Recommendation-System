from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_USER_BGE_M3 = "deepvk/USER-bge-m3"

# Explore cross-dept safety: forbid “repulsive” transitions.
# Default policy is “allow all” except entries below.
DEPT_CROSS_BLACKLIST: dict[str, set[str]] = {
    # Medical content → food/travel is often perceived as tone-deaf.
    "Медицина": {"Еда"},
}


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


_RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_RE_NUM = re.compile(r"\b\d+([.,]\d+)?\b")
_RE_SPACE = re.compile(r"\s+")


def normalize_title_for_dedup(title: str) -> str:
    """
    Aggressive normalization to catch near-duplicates/series like:
    "ПДД 2025" vs "ПДД 2026", "льготы 2026 года", etc.
    """
    t = safe_str(title).lower()
    t = t.replace("ё", "е")
    t = _RE_YEAR.sub(" <year> ", t)
    t = _RE_NUM.sub(" <num> ", t)
    t = re.sub(r"[\"'“”«»()\\[\\]{}:;,.!?/\\\\|—–-]+", " ", t)
    t = _RE_SPACE.sub(" ", t).strip()
    return t


@dataclass(frozen=True)
class ModelArtifacts:
    df: pd.DataFrame
    X: np.ndarray  # [n, d] float32 normalized embeddings
    nn: NearestNeighbors


def load_articles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["article_id"] = df["article_id"].map(safe_str)
    for col in [
        "article_base__title",
        "article_base__department",
        "article_base__rubric",
        "article_flows__primary_flow_name",
        "article_flows__department",
        "article_flows__flow_name_str",
        "article_flows__all_departments_str",
        "article_flows__article_ugc_type",
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

    if "article_base__title" in df.columns:
        df["title_norm"] = df["article_base__title"].map(normalize_title_for_dedup)
    else:
        df["title_norm"] = ""
    return df


def fit_retrieval_model(df: pd.DataFrame) -> ModelArtifacts:
    """
    Retrieval похожести: cosine по эмбеддингам заголовка из `deepvk/USER-bge-m3`.
    Просмотры/лайки/даты НЕ входят в retrieval-вектор — популярность учитывается в rerank
    и только внутри одного департамента (см. `rerank_and_diversify`).
    """
    text_col = "article_base__title"
    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")

    titles = df[text_col].map(safe_str).tolist()
    X = _encode_titles_user_bge_m3(titles)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)
    return ModelArtifacts(df=df, X=X, nn=nn)


def _encode_titles_user_bge_m3(
    titles: list[str],
    *,
    model_name: str = _USER_BGE_M3,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Возвращает L2-нормированные эмбеддинги [n, d] float32.
    SentenceTransformers вернёт numpy при `convert_to_numpy=True`.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `sentence-transformers`.\n"
            "Install: pip install -U sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    emb = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calc_quality_prior(row: pd.Series) -> float:
    """
    Proxy for "quality after click" on current dataset.
    Later should be replaced with readthrough/completion + post-click signals.
    """
    like_rate = row.get("article_stats__like_rate", np.nan)
    comment_rate = row.get("article_stats__comment_rate", np.nan)
    favs = row.get("article_stats__stats_favorites", np.nan)
    views = row.get("article_stats__stats_views", np.nan)

    lr = float(like_rate) if pd.notna(like_rate) else 0.0
    cr = float(comment_rate) if pd.notna(comment_rate) else 0.0
    fav_rate = (float(favs) / max(float(views), 1.0)) if pd.notna(favs) and pd.notna(views) else 0.0

    # compress long tails
    return float(_sigmoid(12.0 * (0.7 * lr + 0.3 * fav_rate) + 4.0 * cr))


def _calc_trending_prior(row: pd.Series) -> float:
    views = row.get("article_stats__stats_views", np.nan)
    days = row.get("article_dates__days_since_published", np.nan)
    v = float(views) if pd.notna(views) else 0.0
    d = float(days) if pd.notna(days) else 365.0
    # simple decay: more views + fresher => higher
    return float(math.log1p(v) / math.sqrt(d + 1.0))


def _calc_freshness(row: pd.Series) -> float:
    days = row.get("article_dates__days_since_published", np.nan)
    d = float(days) if pd.notna(days) else 365.0
    # 1.0 for today, decays with days
    return float(1.0 / (1.0 + d / 30.0))


def rerank_and_diversify(
    df: pd.DataFrame,
    source_idx: int,
    cand_indices: np.ndarray,
    cand_sims: np.ndarray,
    *,
    k: int,
    explore_slots: int = 2,
    max_same_author: int = 2,
    max_same_rubric: int = 4,
    avoid_same_title_norm: bool = True,
    w_sim: float = 0.72,
    w_quality: float = 0.18,
    w_trend: float = 0.06,
    w_fresh: float = 0.04,
    explore_gap: int = 5,
    core_similar: int = 1,
    explore_sample_pool: int = 300,
    explore_temperature: float = 0.35,
) -> list[dict]:
    src = df.iloc[source_idx]
    src_dept = safe_str(src.get("article_base__department", ""))
    src_rubric = safe_str(src.get("article_base__rubric", ""))
    src_title_norm = safe_str(src.get("title_norm", ""))
    src_id = safe_str(src["article_id"])

    # Build candidate table
    crows = df.iloc[cand_indices].copy()
    crows["sim"] = cand_sims
    crows["quality"] = crows.apply(_calc_quality_prior, axis=1)
    crows["trend"] = crows.apply(_calc_trending_prior, axis=1)
    crows["fresh"] = crows.apply(_calc_freshness, axis=1)
    crows["cand_dept"] = crows["article_base__department"].map(safe_str)

    # Gate: enforce topicality as primary driver
    # (keep enough candidates; threshold is soft and depends on distribution)
    crows = crows[crows["sim"] >= max(0.08, float(np.quantile(crows["sim"].values, 0.5)))].copy()
    if len(crows) == 0:
        crows = df.iloc[cand_indices].copy()
        crows["sim"] = cand_sims
        crows["quality"] = crows.apply(_calc_quality_prior, axis=1)
        crows["trend"] = crows.apply(_calc_trending_prior, axis=1)
        crows["fresh"] = crows.apply(_calc_freshness, axis=1)
        crows["cand_dept"] = crows["article_base__department"].map(safe_str)

    def _score_within_dept(sub: pd.DataFrame) -> pd.DataFrame:
        """Популярность/качество — только ранги внутри подвыборки (один департамент)."""
        sub = sub.copy()
        if len(sub) == 0:
            return sub
        if len(sub) == 1:
            sub["qual_d"] = 0.5
            sub["trend_d"] = 0.5
        else:
            sub["qual_d"] = sub["quality"].rank(pct=True)
            sub["trend_d"] = sub["trend"].rank(pct=True)
        sub["score"] = (
            w_sim * sub["sim"]
            + w_quality * sub["qual_d"]
            + w_trend * sub["trend_d"]
            + w_fresh * sub["fresh"]
        )
        if avoid_same_title_norm and src_title_norm:
            sub.loc[sub.get("title_norm", "").astype(str) == src_title_norm, "score"] *= 0.20
        return sub.sort_values("score", ascending=False).reset_index(drop=True)

    # Сначала кандидаты того же департамента (тема + популярность внутри отдела); потом остальные — только similarity
    if src_dept:
        sd_raw = crows[crows["cand_dept"] == src_dept]
        if len(sd_raw) > 0:
            sd = _score_within_dept(sd_raw)
            od = crows[crows["cand_dept"] != src_dept].copy()
            if len(od):
                od["score"] = w_sim * od["sim"] * 0.35
                od = od.sort_values("score", ascending=False).reset_index(drop=True)
            crows_ordered = pd.concat([sd, od], ignore_index=True) if len(od) else sd
        else:
            # в узком пуле kNN нет соседей из того же департамента — скорим весь пул (ранги популярности по нему)
            crows_ordered = _score_within_dept(crows)
    else:
        crows_ordered = _score_within_dept(crows)

    score_by_aid = {
        safe_str(aid): float(sc) for aid, sc in zip(crows_ordered["article_id"], crows_ordered["score"])
    }

    # Greedy selection with diversity constraints + explore slot(s)
    picked: list[dict] = []
    used_title_norm: set[str] = {src_title_norm} if src_title_norm else set()
    author_cnt: dict[str, int] = {}
    rubric_cnt: dict[str, int] = {}

    def can_pick(r: pd.Series) -> bool:
        aid = safe_str(r.get("article_base__author_id", ""))
        rub = safe_str(r.get("article_base__rubric", ""))
        tnorm = safe_str(r.get("title_norm", ""))
        if avoid_same_title_norm and tnorm and tnorm in used_title_norm:
            return False
        if aid and author_cnt.get(aid, 0) >= max_same_author:
            return False
        if rub and rubric_cnt.get(rub, 0) >= max_same_rubric:
            return False
        return True

    # Для explore не сравниваем title_norm с уже взятыми similar: иначе второй explore часто
    # отваливается (те же шаблоны интервью/рубрики), хотя это другие статьи.
    explore_title_norms: set[str] = set()

    def can_pick_explore(r: pd.Series, *, picked_ids: set[str]) -> bool:
        """
        Для explore: без капов автор/рубрика; title_norm — только как у источника или
        дубликат между двумя explore-слотами, не как у similar.
        """
        cid = safe_str(r.get("article_id", ""))
        if not cid or cid in picked_ids:
            return False
        tnorm = safe_str(r.get("title_norm", ""))
        if avoid_same_title_norm:
            if src_title_norm and tnorm == src_title_norm:
                return False
            if tnorm and tnorm in explore_title_norms:
                return False
        return True

    # Reserve slots for explore; similar fills the rest (but not more than k - explore_slots)
    explore_slots = min(max(0, explore_slots), k)
    target_sim_slots = max(0, k - explore_slots)
    similar_picked: list[dict] = []

    # Core similar: guarantee at least a few very high-sim items even if their final score is low.
    # This helps keep the carousel anchored to the source topic under diversity caps.
    core_similar = max(0, int(core_similar))
    if core_similar > 0:
        # Important: pick globally top by `sim` (baseline-style), without same-dept preference.
        core_pool = crows_ordered.sort_values("sim", ascending=False)
        for _, r in core_pool.iterrows():
            if len(similar_picked) >= min(core_similar, target_sim_slots):
                break
            cid = safe_str(r.get("article_id", ""))
            if not cid:
                continue
            # minimal dedup only
            tnorm = safe_str(r.get("title_norm", ""))
            if avoid_same_title_norm and tnorm and tnorm in used_title_norm:
                continue
            similar_picked.append(
                {
                    "source_article_id": src_id,
                    "candidate_article_id": cid,
                    "score": float(r.get("score", 0.0)),
                    "similarity": float(r.get("sim", 0.0)),
                    "mix": "similar",
                    "candidate_title": safe_str(r.get("article_base__title", "")),
                    "candidate_department": safe_str(r.get("article_base__department", "")),
                    "candidate_rubric": safe_str(r.get("article_base__rubric", "")),
                }
            )
            if tnorm:
                used_title_norm.add(tnorm)
            aid = safe_str(r.get("article_base__author_id", ""))
            if aid:
                author_cnt[aid] = author_cnt.get(aid, 0) + 1
            rub = safe_str(r.get("article_base__rubric", ""))
            if rub:
                rubric_cnt[rub] = rubric_cnt.get(rub, 0) + 1

    for _, r in crows_ordered.iterrows():
        if len(similar_picked) >= target_sim_slots:
            break
        cid0 = safe_str(r.get("article_id", ""))
        if cid0 and cid0 in {x.get("candidate_article_id", "") for x in similar_picked}:
            continue
        if not can_pick(r):
            continue
        similar_picked.append(
            {
                "source_article_id": src_id,
                "candidate_article_id": safe_str(r["article_id"]),
                "score": float(r["score"]),
                "similarity": float(r["sim"]),
                "mix": "similar",
                "candidate_title": safe_str(r.get("article_base__title", "")),
                "candidate_department": safe_str(r.get("article_base__department", "")),
                "candidate_rubric": safe_str(r.get("article_base__rubric", "")),
            }
        )
        tnorm = safe_str(r.get("title_norm", ""))
        if tnorm:
            used_title_norm.add(tnorm)
        aid = safe_str(r.get("article_base__author_id", ""))
        if aid:
            author_cnt[aid] = author_cnt.get(aid, 0) + 1
        rub = safe_str(r.get("article_base__rubric", ""))
        if rub:
            rubric_cnt[rub] = rubric_cnt.get(rub, 0) + 1

    picked = list(similar_picked)

    # Explore: must be candidates NOT already chosen as similar — otherwise top explore_score
    # rows are often duplicates of similar picks and can_pick rejects everything → 0 explore.
    if explore_slots > 0:
        explore_picked: list[dict] = []
        picked_ids: set[str] = {safe_str(x["candidate_article_id"]) for x in picked}
        exp = crows.copy()
        exp = exp[~exp["article_id"].astype(str).map(safe_str).isin(picked_ids)]
        if src_dept:
            exp_same = exp[exp["cand_dept"] == src_dept].copy()
            exp_other = exp[exp["cand_dept"] != src_dept].copy()
            frames: list[pd.DataFrame] = []
            for sub, weight in ((exp_same, 1.0), (exp_other, 0.35)):
                if len(sub) == 0:
                    continue
                if len(sub) == 1:
                    sub["qual_d"] = 0.5
                    sub["trend_d"] = 0.5
                else:
                    sub["qual_d"] = sub["quality"].rank(pct=True)
                    sub["trend_d"] = sub["trend"].rank(pct=True)
                sub["explore_score"] = weight * (0.6 * sub["qual_d"] + 0.4 * sub["trend_d"])
                frames.append(sub.sort_values("explore_score", ascending=False))
            exp = pd.concat(frames, ignore_index=True) if frames else exp.iloc[0:0]
        else:
            exp["qual_d"] = exp["quality"].rank(pct=True)
            exp["trend_d"] = exp["trend"].rank(pct=True)
            exp["explore_score"] = 0.6 * exp["qual_d"] + 0.4 * exp["trend_d"]
            exp = exp.sort_values("explore_score", ascending=False)
        explore_added = 0
        explore_same_target = max(0, explore_slots - 1) if src_dept else explore_slots
        explore_cross_target = 1 if src_dept and explore_slots > 0 else 0
        explore_same_added = 0
        explore_cross_added = 0

        def _try_append_explore(r: pd.Series) -> bool:
            nonlocal explore_added, explore_same_added, explore_cross_added
            if explore_added >= explore_slots:
                return False
            if not can_pick_explore(r, picked_ids=picked_ids):
                return False

            cand_dept = safe_str(r.get("article_base__department", ""))
            cand_rub = safe_str(r.get("article_base__rubric", ""))

            if src_dept:
                if cand_dept == src_dept:
                    if explore_same_added >= explore_same_target:
                        return False
                    # Explore-in-dept should broaden within dept: prefer other rubric.
                    if src_rubric and cand_rub and cand_rub == src_rubric:
                        return False
                    explore_same_added += 1
                else:
                    if explore_cross_added >= explore_cross_target:
                        return False
                    forbidden = DEPT_CROSS_BLACKLIST.get(src_dept, set())
                    if cand_dept and cand_dept in forbidden:
                        return False
                    explore_cross_added += 1

            cid = safe_str(r["article_id"])
            explore_picked.append(
                {
                    "source_article_id": src_id,
                    "candidate_article_id": cid,
                    "score": float(score_by_aid.get(cid, w_sim * float(r["sim"]))),
                    "similarity": float(r["sim"]),
                    "mix": "explore",
                    "candidate_title": safe_str(r.get("article_base__title", "")),
                    "candidate_department": safe_str(r.get("article_base__department", "")),
                    "candidate_rubric": safe_str(r.get("article_base__rubric", "")),
                }
            )
            explore_added += 1
            picked_ids.add(cid)
            tnorm = safe_str(r.get("title_norm", ""))
            if tnorm:
                explore_title_norms.add(tnorm)
            return True

        for _, r in exp.iterrows():
            if explore_added >= explore_slots:
                break
            _try_append_explore(r)

        # Fallback: после гейта по sim в `crows` мало строк — добираем explore из всех соседей kNN
        if explore_added < explore_slots:
            fb = df.iloc[cand_indices].copy()
            fb["sim"] = cand_sims
            fb["quality"] = fb.apply(_calc_quality_prior, axis=1)
            fb["trend"] = fb.apply(_calc_trending_prior, axis=1)
            fb["cand_dept"] = fb["article_base__department"].map(safe_str)
            fb = fb[~fb["article_id"].astype(str).map(safe_str).isin(picked_ids)]
            if src_dept:
                same = fb[fb["cand_dept"] == src_dept].sort_values("trend", ascending=False)
                other = fb[fb["cand_dept"] != src_dept].sort_values("trend", ascending=False)
                fb = pd.concat([same, other], ignore_index=True)
            else:
                fb = fb.sort_values("trend", ascending=False)
            for _, r in fb.iterrows():
                if explore_added >= explore_slots:
                    break
                _try_append_explore(r)

        # If explore is too repetitive across sources, sample from top-M by explore_score
        # using a deterministic seed derived from the source id.
        if len(explore_picked) > 0 and explore_sample_pool > 1 and explore_temperature > 0:
            seed = int.from_bytes(
                hashlib.blake2b(src_id.encode("utf-8"), digest_size=8).digest(), "little"
            )
            rng = np.random.default_rng(seed)

            # Rebuild a candidate pool excluding already picked ids and resample explore picks.
            exp2 = exp.copy()
            exp2 = exp2[~exp2["article_id"].astype(str).map(safe_str).isin({x["candidate_article_id"] for x in picked})]
            exp2 = exp2.head(int(explore_sample_pool)).copy()
            if "explore_score" not in exp2.columns:
                exp2["explore_score"] = 0.0

            # Reset counters and resample
            explore_picked = []
            explore_added = 0
            explore_same_added = 0
            explore_cross_added = 0
            explore_title_norms = set()

            def _sample_row(df_pool: pd.DataFrame) -> tuple[int, pd.Series] | None:
                if len(df_pool) == 0:
                    return None
                w = pd.to_numeric(df_pool["explore_score"], errors="coerce").fillna(0.0).astype(float).values
                # softmax with temperature
                z = (w - float(np.max(w))) / float(explore_temperature)
                p = np.exp(np.clip(z, -30, 30))
                p = p / max(float(p.sum()), 1e-12)
                idx = int(rng.choice(len(df_pool), p=p))
                return idx, df_pool.iloc[idx]

            pool = exp2.reset_index(drop=True)
            # try more attempts than slots to handle rejections
            attempts = 0
            while explore_added < explore_slots and len(pool) > 0 and attempts < explore_slots * 50:
                attempts += 1
                sampled = _sample_row(pool)
                if sampled is None:
                    break
                idx, r = sampled
                pool = pool.drop(pool.index[idx]).reset_index(drop=True)
                _try_append_explore(r)

        # Final fallback: if strict quotas (same-dept + cross-dept) can't be satisfied,
        # relax progressively to always return exactly `k` recommendations.
        if explore_added < explore_slots:
            # (1) try again with deterministic scan of full `exp`
            for _, r in exp.iterrows():
                if explore_added >= explore_slots:
                    break
                _try_append_explore(r)

        if explore_added < explore_slots and src_dept and src_rubric:
            # (2) last resort: allow same-rubric within dept to fill remaining slots
            for _, r in exp.iterrows():
                if explore_added >= explore_slots:
                    break
                cand_dept = safe_str(r.get("article_base__department", ""))
                cand_rub = safe_str(r.get("article_base__rubric", ""))
                if cand_dept == src_dept and cand_rub == src_rubric:
                    rr = r.copy()
                    rr["article_base__rubric"] = ""
                    _try_append_explore(rr)

        # Interleave explore into the final carousel instead of putting it at the end.
        # Default: 1 explore per ~5 items (tunable via explore_gap), with the last explore at position k.
        gap = max(2, int(explore_gap))
        if explore_slots > 0:
            # 1-based positions where we *prefer* to place explore.
            raw_pos = [min(k, gap * i) for i in range(1, explore_slots)]
            raw_pos.append(k)
            explore_positions = sorted({p for p in raw_pos if 1 <= p <= k})
        else:
            explore_positions = []

        merged: list[dict] = []
        sim_i = 0
        exp_i = 0
        for pos in range(1, k + 1):
            want_explore = pos in explore_positions and exp_i < len(explore_picked)
            if want_explore:
                merged.append(explore_picked[exp_i])
                exp_i += 1
            elif sim_i < len(similar_picked):
                merged.append(similar_picked[sim_i])
                sim_i += 1
            elif exp_i < len(explore_picked):
                merged.append(explore_picked[exp_i])
                exp_i += 1
            else:
                break

        # Assign rank as carousel position (order), not by score.
        for i, row in enumerate(merged, start=1):
            row["rank"] = i
        return merged[:k]

    # No explore: rank is just the similar order.
    for i, row in enumerate(picked[:k], start=1):
        row["rank"] = i
    return picked[:k]


def build_recommendations(
    art: ModelArtifacts,
    *,
    k: int,
    candidate_pool: int,
    explore_slots: int,
    batch_size: int,
) -> pd.DataFrame:
    df = art.df
    X = art.X
    nn = art.nn

    results: list[dict] = []
    n = df.shape[0]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        distances, indices = nn.kneighbors(X[start:end], n_neighbors=min(candidate_pool + 1, n))
        # cosine distance => similarity = 1 - distance
        sims = 1.0 - distances

        for i in range(end - start):
            src_idx = start + i
            src_id = safe_str(df.iloc[src_idx]["article_id"])
            if not src_id:
                continue

            cand_idx = indices[i]
            cand_sim = sims[i]

            # remove self
            mask = cand_idx != src_idx
            cand_idx = cand_idx[mask]
            cand_sim = cand_sim[mask]

            picked = rerank_and_diversify(
                df,
                src_idx,
                cand_idx,
                cand_sim,
                k=k,
                explore_slots=explore_slots,
            )
            results.extend(picked)

    out = pd.DataFrame(results)
    if not out.empty:
        # `rerank_and_diversify` assigns `rank` as the final carousel position.
        out["rank"] = pd.to_numeric(out.get("rank", np.nan), errors="coerce")
        # Fallback if older artifacts are mixed in
        if out["rank"].isna().any():
            out["rank"] = out.groupby("source_article_id").cumcount() + 1
        out = out.sort_values(["source_article_id", "rank"], ascending=[True, True]).reset_index(drop=True)
    return out


def build_recommendations_to_csv(
    art: ModelArtifacts,
    *,
    out_path: str,
    k: int,
    candidate_pool: int,
    explore_slots: int,
    batch_size: int,
) -> None:
    """
    Streaming writer version for large corpora: writes incrementally to CSV.
    """
    df = art.df
    X = art.X
    nn = art.nn

    n = df.shape[0]
    wrote_rows = 0

    header_written = False
    columns = [
        "source_article_id",
        "candidate_article_id",
        "rank",
        "score",
        "similarity",
        "mix",
        "candidate_title",
        "candidate_department",
        "candidate_rubric",
    ]

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        distances, indices = nn.kneighbors(X[start:end], n_neighbors=min(candidate_pool + 1, n))
        sims = 1.0 - distances  # cosine distance => similarity

        batch_rows: list[dict] = []
        for i in range(end - start):
            src_idx = start + i
            src_id = safe_str(df.iloc[src_idx]["article_id"])
            if not src_id:
                continue

            cand_idx = indices[i]
            cand_sim = sims[i]

            mask = cand_idx != src_idx
            cand_idx = cand_idx[mask]
            cand_sim = cand_sim[mask]

            picked = rerank_and_diversify(
                df,
                src_idx,
                cand_idx,
                cand_sim,
                k=k,
                explore_slots=explore_slots,
            )
            for r, row in enumerate(picked, start=1):
                row["rank"] = r
                batch_rows.append(row)

        if batch_rows:
            out_df = pd.DataFrame(batch_rows)
            # keep stable schema for downstream consumption
            for c in columns:
                if c not in out_df.columns:
                    out_df[c] = ""
            out_df = out_df[columns]
            out_df.to_csv(out_path, mode="a", index=False, header=not header_written)
            header_written = True
            wrote_rows += len(out_df)

        print(f"processed {end:,}/{n:,} articles | wrote {wrote_rows:,} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build improved i2i recommendations from tj_article.csv")
    ap.add_argument("--data", default="tj_article.csv", help="Input articles CSV (sep=';')")
    ap.add_argument("--out", default="recs_i2i.csv", help="Output CSV path")
    ap.add_argument("--k", type=int, default=12, help="Final recommendations per article")
    ap.add_argument("--candidate-pool", type=int, default=200, help="Initial neighbor pool size")
    ap.add_argument("--explore-slots", type=int, default=2, help="How many explore slots to mix in")
    ap.add_argument("--batch-size", type=int, default=2000, help="KNN batch size")
    args = ap.parse_args()

    df = load_articles(args.data)
    art = fit_retrieval_model(df)
    # stream to CSV to avoid holding everything in memory
    build_recommendations_to_csv(
        art,
        out_path=args.out,
        k=args.k,
        candidate_pool=args.candidate_pool,
        explore_slots=args.explore_slots,
        batch_size=args.batch_size,
    )
    print(f"Done. Output at {args.out}")


if __name__ == "__main__":
    main()

