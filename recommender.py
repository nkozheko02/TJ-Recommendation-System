from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.neighbors import NearestNeighbors

import numpy as np

_USER_BGE_M3 = "deepvk/USER-bge-m3"


def _safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


@dataclass(frozen=True)
class RecommenderArtifacts:
    df: pd.DataFrame
    article_id_to_row: dict[str, int]
    nn: NearestNeighbors
    X: np.ndarray  # normalized title embeddings


def load_articles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Normalize IDs and a few frequently-used string columns.
    df["article_id"] = df["article_id"].map(_safe_str)
    for col in [
        "article_base__title",
        "article_flows__flow_name_str",
        "article_flows__all_departments_str",
        "article_base__department",
        "article_base__rubric",
        "article_flows__primary_flow_name",
        "article_flows__department",
        "article_flows__article_ugc_type",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_safe_str)

    # Robust numeric conversion for columns that may use comma as decimal separator.
    # (We only convert columns that are plausibly numeric to avoid mangling free-text fields.)
    numeric_prefixes = ("article_stats__", "article_dates__", "article_author__")
    for col in df.columns:
        if not col.startswith(numeric_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


def fit_recommender(df: pd.DataFrame) -> RecommenderArtifacts:
    """kNN по семантическим эмбеддингам заголовка (USER-bge-m3)."""
    text_col = "article_base__title"
    if text_col not in df.columns:
        raise ValueError(f"Missing required column {text_col!r}")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `sentence-transformers`.\n"
            "Install: pip install -U sentence-transformers"
        ) from e

    model = SentenceTransformer(_USER_BGE_M3)
    titles = df[text_col].map(_safe_str).tolist()
    X = model.encode(
        titles,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    article_id_to_row = {aid: i for i, aid in enumerate(df["article_id"].tolist()) if aid}
    return RecommenderArtifacts(df=df, article_id_to_row=article_id_to_row, nn=nn, X=X)


def recommend(
    art: RecommenderArtifacts,
    article_id: str,
    k: int = 10,
    *,
    same_department_boost: bool = True,
) -> pd.DataFrame:
    article_id = _safe_str(article_id)
    if article_id not in art.article_id_to_row:
        raise KeyError(f"Unknown article_id: {article_id}")

    row = art.article_id_to_row[article_id]
    distances, indices = art.nn.kneighbors(art.X[row], n_neighbors=min(k + 50, art.df.shape[0]))
    distances = distances.ravel()
    indices = indices.ravel()

    src = art.df.iloc[row]
    src_dept = _safe_str(src.get("article_base__department", ""))

    recs = []
    for dist, idx in zip(distances, indices):
        if idx == row:
            continue
        cand = art.df.iloc[idx]
        cand_id = _safe_str(cand["article_id"])
        if not cand_id:
            continue

        score = float(1.0 - dist)  # cosine similarity
        if same_department_boost and src_dept and _safe_str(cand.get("article_base__department", "")) == src_dept:
            score *= 1.05

        recs.append(
            {
                "source_article_id": article_id,
                "candidate_article_id": cand_id,
                "candidate_title": _safe_str(cand.get("article_base__title", "")),
                "candidate_department": _safe_str(cand.get("article_base__department", "")),
                "candidate_rubric": _safe_str(cand.get("article_base__rubric", "")),
                "similarity": score,
            }
        )
        if len(recs) >= k:
            break

    return pd.DataFrame(recs).sort_values("similarity", ascending=False, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple content-based recommender over tj_article.csv")
    ap.add_argument("--data", default="tj_article.csv", help="Path to tj_article.csv")
    ap.add_argument("--article-id", required=True, help="Article UUID to recommend for")
    ap.add_argument("--k", type=int, default=10, help="Number of recommendations")
    args = ap.parse_args()

    df = load_articles(args.data)
    art = fit_recommender(df)
    out = recommend(art, args.article_id, k=args.k)
    pd.set_option("display.max_colwidth", 120)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

