"""Примеры выдач для двух нейтральных статей-источников:
- baseline: cosine kNN top-12 без LTR/explore/правил;
- ours: полный production-пайплайн (similar+explore → LTR → rerank + интерливинг 4/7/10).

Источники выбираются из департаментов про деньги/путешествия/быт; всё, что про
болезни, смерти и прочие неприятные темы — отсекается.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from serve_carousel import build_context, serve_one
from train_ltr_full import REPORTS_DIR, safe_str


# --- настройки -------------------------------------------------------------

EXAMPLES_DIR = REPORTS_DIR / "examples_neutral"
EXAMPLES_DIR.mkdir(exist_ok=True)

K = 12

# Темы, которые сразу выкидываем как «не нейтральные».
BLOCKED_DEPTS = {
    "Медицина",
    "Здоровье",
    "Психология",
}
BLOCKED_WORDS = [
    "болезн",
    "смерт",
    "умер",
    "рак",
    "covid",
    "вирус",
    "похорон",
    "развод",
    "увольн",
    "долг",
    "кредит",
    "ипотек",
    "штраф",
    "налог",
    "суд",
    "мошенн",
    "обман",
    "афер",
    "психол",
    "тревож",
    "депресс",
    "сектант",
    "военн",
    "оруж",
    "медкомис",
    "медицин",
    "инвалид",
    "врач",
    "клиник",
    "поликлиник",
    "донор",
    "лечен",
    "росздрав",
    "дмс",
    "алкоголь",
    "алкого",
    "бросил пить",
    "не пить",
    "пьянств",
    "запой",
    "зож",
    "менопауз",
    "метабол",
    "ожирен",
    "лишний вес",
    "лиш­ний вес",
    "похуд",
    "выгоран",
    "стресс",
    "текучест",
]

# Темы, которые наоборот хорошо подходят как нейтральные источники.
# Названия департаментов соответствуют реальной таксономии Т—Ж.
PREFERRED_DEPTS = {"Технологии", "Еда", "Чемодан", "Поп-культура", "Авто", "Город", "Спорт и фитнес", "Животные"}


def is_neutral(row: pd.Series) -> bool:
    dept = safe_str(row.get("article_base__department", ""))
    rub = safe_str(row.get("article_base__rubric", "")).lower()
    title = safe_str(row.get("article_base__title", "")).lower()
    if dept in BLOCKED_DEPTS:
        return False
    blob = f"{title} {rub}"
    for w in BLOCKED_WORDS:
        if w in blob:
            return False
    return True


def baseline_topk(ctx, src_idx: int, k: int = K) -> pd.DataFrame:
    """Чистый cosine kNN top-K без LTR и без продуктовых правил.

    Возвращает таблицу с ПОЛНЫМИ заголовками — никакой обрезки.
    """
    art = ctx.art
    dist, idx = art.nn.kneighbors(art.X[src_idx : src_idx + 1], n_neighbors=k + 1)
    idx = idx[0]
    sim = 1.0 - dist[0]
    mask = idx != src_idx
    idx = idx[mask][:k].astype(np.int64)
    sim = sim[mask][:k].astype(np.float32)
    rows = []
    for r, (j, s) in enumerate(zip(idx.tolist(), sim.tolist()), start=1):
        row = art.df.iloc[int(j)]
        rows.append({
            "rank": r,
            "mix": "baseline",
            "title": safe_str(row.get("article_base__title", "")),
            "department": safe_str(row.get("article_base__department", "")),
            "rubric": safe_str(row.get("article_base__rubric", "")),
            "age_days": int(row.get("article_dates__days_since_published", 0) or 0),
            "views": int(row.get("article_stats__stats_views", 0) or 0),
            "sim": round(float(s), 3),
        })
    return pd.DataFrame(rows)


def our_pipeline(ctx, source_id: str) -> pd.DataFrame:
    """Полная карусель K=12 = 9 similar + 3 explore с интерливингом 4/7/10.

    Заголовки в выдаче — без обрезки.
    """
    out = serve_one(ctx, source_id).copy()
    cols = ["rank", "mix", "candidate_title", "candidate_department", "candidate_rubric", "similarity", "score"]
    out = out[cols].rename(columns={
        "candidate_title": "title",
        "candidate_department": "department",
        "candidate_rubric": "rubric",
        "similarity": "sim",
        "score": "ltr_score",
    })
    out["sim"] = out["sim"].round(3)
    out["ltr_score"] = out["ltr_score"].round(3)
    return out


UNWANTED_EXPLORE_TITLES = {
    "бесят тревожные люди",
}
# Префиксы, отсеивающие explore-статьи в «жалобном» стиле — пользователь
# просил избегать подобных «бесят/бесит ...» материалов в дополнительных
# источниках.
UNWANTED_EXPLORE_PREFIXES = ("бесят ", "бесит ", "бесит,", "бесят,")


def extra_source_ok(ctx, source_id: str) -> bool:
    """Проверяет, что у источника карусель ровно 12 элементов, в explore нет
    нежелательных «жалобных» статей и весь top-12 проходит фильтр нейтральности
    (никаких медицинских/судебных/долговых тем в заголовках similar и explore).

    Используется для отбора дополнительных источников (5-го, 6-го и т.д.) —
    чтобы в их выдаче не повторялась «Бесят тревожные люди» и не вылезали
    тяжёлые темы.
    """
    try:
        out = serve_one(ctx, source_id)
    except Exception:
        return False
    if len(out) != 12:
        return False
    explore_titles = [
        safe_str(t).strip().lower()
        for t in out.loc[out["mix"] == "explore", "candidate_title"].tolist()
    ]
    if any(t in UNWANTED_EXPLORE_TITLES for t in explore_titles):
        return False
    if any(t.startswith(UNWANTED_EXPLORE_PREFIXES) for t in explore_titles):
        return False
    for t in out["candidate_title"].tolist():
        blob = safe_str(t).lower()
        for w in BLOCKED_WORDS:
            if w in blob:
                return False
    return True


def find_neutral_sources(ctx) -> list[str]:
    """Возвращает шесть источников из разных тематических кластеров:
      1) Технологии — фиксированный id (та же статья, что и в первой версии примера),
      2) Подарки / Еда — фиксированный id,
      3) Путешествия — самая популярная статья из департамента,
      4) Еда / Дом / Развлечения — самая популярная статья (без повтора с п.2),
      5-6) Два дополнительных нейтральных источника, у которых в explore нет
           статьи «Бесят тревожные люди» (чтобы показать, что софт-макс
           действительно даёт разнообразие между источниками).
    """

    art = ctx.art
    df = art.df.copy()
    df["views"] = pd.to_numeric(df.get("article_stats__stats_views", 0), errors="coerce").fillna(0)
    df["age"] = pd.to_numeric(df.get("article_dates__days_since_published", 365), errors="coerce").fillna(365)
    df = df[df.apply(is_neutral, axis=1)].copy()

    coview_sources = set(ctx.coview_neigh_idx.keys())
    id_to_idx = art.article_id_to_row

    def has_coview(aid: str) -> bool:
        idx = id_to_idx.get(safe_str(aid))
        return idx is not None and idx in coview_sources

    df = df[df["article_id"].map(has_coview)]

    chosen: list[str] = []
    used: set[str] = set()

    def pick_top(dept_names: list[str], age_min: int = 14, age_max: int = 1200) -> str | None:
        pool = df[df["article_base__department"].isin(dept_names)]
        pool = pool[pool["age"].between(age_min, age_max)]
        pool = pool.sort_values("views", ascending=False)
        for _, row in pool.head(150).iterrows():
            aid = safe_str(row["article_id"])
            if aid not in used:
                return aid
        return None

    # 1) tech — фиксированная статья про смартфоны до 20 000
    tech_fixed = "c6a76dd8-4022-441a-be6f-03b96fb709e4"
    if tech_fixed in id_to_idx and tech_fixed in [safe_str(x) for x in df["article_id"].values]:
        chosen.append(tech_fixed)
    else:
        aid = pick_top(["Технологии"])
        if aid:
            chosen.append(aid)
    used.update(chosen)

    # 2) подарки / еда — фиксированная статья
    gifts_fixed = "247bd1d2-cfe5-4722-8976-870960a1ee5f"
    if gifts_fixed in id_to_idx and gifts_fixed in [safe_str(x) for x in df["article_id"].values]:
        chosen.append(gifts_fixed)
        used.add(gifts_fixed)
    else:
        aid = pick_top(["Еда", "Дом"])
        if aid:
            chosen.append(aid)
            used.add(aid)

    # 3) путешествия — в таксономии Т—Ж департамент называется «Чемодан»
    aid = pick_top(["Чемодан"])
    if aid:
        chosen.append(aid)
        used.add(aid)

    # 4) поп-культура / авто / спорт / город — другая тема, без повторов
    aid = pick_top(["Поп-культура", "Авто", "Спорт и фитнес", "Город"])
    if aid:
        chosen.append(aid)
        used.add(aid)

    # 5) ещё один нейтральный источник, в выдаче которого нет «Бесят тревожные
    # люди» и подобных «жалобных» материалов. Берём из департаментов, ещё не
    # представленных в предыдущих четырёх (иначе будет дубль темы).
    used_depts = {
        safe_str(df.loc[df["article_id"] == aid, "article_base__department"].iloc[0])
        for aid in chosen
        if not df.loc[df["article_id"] == aid].empty
    }
    # Для дополнительных источников (5-го и 6-го) берём «лёгкие» темы по
    # порядку приоритета: сначала «Животные» (котики, собаки — обычно
    # безопасно), потом «Дом» (рецепты, мебель), «Спорт и фитнес» и в самом
    # конце «Авто». Каждый раз департамент должен быть новым (без повторов).
    def pick_extra_source(used_depts_set: set[str]) -> str | None:
        priority = [
            d for d in ["Животные", "Дом", "Спорт и фитнес", "Авто"]
            if d not in used_depts_set
        ]
        for dept in priority:
            extra_pool = df[df["article_base__department"] == dept]
            extra_pool = extra_pool[extra_pool["age"].between(14, 1200)]
            extra_pool = extra_pool.sort_values("views", ascending=False)
            for _, row in extra_pool.head(400).iterrows():
                aid = safe_str(row["article_id"])
                if aid in used:
                    continue
                if not extra_source_ok(ctx, aid):
                    continue
                return aid
        return None

    # 5-й источник
    aid5 = pick_extra_source(used_depts)
    if aid5:
        chosen.append(aid5)
        used.add(aid5)
        dept5 = safe_str(df.loc[df["article_id"] == aid5, "article_base__department"].iloc[0])
        used_depts = used_depts | {dept5}

    # 6-й источник — из другого «лёгкого» департамента
    aid6 = pick_extra_source(used_depts)
    if aid6:
        chosen.append(aid6)
        used.add(aid6)

    return chosen[:6]


def render_table(df: pd.DataFrame, title: str) -> str:
    """Аккуратное текстовое представление таблицы для печати."""
    lines = [f"### {title}"]
    with pd.option_context("display.max_colwidth", 64, "display.width", 200, "display.colheader_justify", "left"):
        lines.append(df.to_string(index=False))
    return "\n".join(lines)


def main() -> None:
    print("[serve] build context…")
    ctx = build_context()
    art = ctx.art

    sources = find_neutral_sources(ctx)
    if not sources:
        print("не нашли подходящих нейтральных источников")
        return
    print("источники:", sources)

    all_text: list[str] = []
    for sid in sources:
        src_idx = art.article_id_to_row.get(sid)
        if src_idx is None:
            print(f"skip: {sid} нет в индексе")
            continue
        src_row = art.df.iloc[src_idx]
        src_title = safe_str(src_row.get("article_base__title", ""))
        src_dept = safe_str(src_row.get("article_base__department", ""))
        src_rub = safe_str(src_row.get("article_base__rubric", ""))

        header = (
            f"\n========================================================================\n"
            f"ИСТОЧНИК: «{src_title}»\n"
            f"  департамент: {src_dept} | рубрика: {src_rub} | article_id: {sid}\n"
            f"========================================================================"
        )
        print(header)
        all_text.append(header)

        base_df = baseline_topk(ctx, src_idx, k=K)
        ours_df = our_pipeline(ctx, sid)

        base_path = EXAMPLES_DIR / f"baseline_{sid}.csv"
        ours_path = EXAMPLES_DIR / f"ours_{sid}.csv"
        base_df.to_csv(base_path, index=False)
        ours_df.to_csv(ours_path, index=False)

        t1 = render_table(base_df, "BASELINE — cosine kNN top-12 (без LTR и правил)")
        t2 = render_table(ours_df, "OURS — similar + explore → LTR → rerank, интерливинг 4/7/10")
        print(t1)
        print(t2)
        all_text.extend([t1, t2])
        print(f"saved: {base_path}\nsaved: {ours_path}")

    (EXAMPLES_DIR / "report.txt").write_text("\n\n".join(all_text), encoding="utf-8")
    print(f"\nfull text saved: {EXAMPLES_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
