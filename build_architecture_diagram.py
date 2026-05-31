"""Архитектурная диаграмма рекомендательной системы Т-Ж.

Простой заголовок, лаконичные блоки, длинный текст не выходит наружу.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / "recs_system_diagram.png"
OUT_SVG = ROOT / "recs_system_diagram.svg"


COLORS = {
    "data": "#fff3d6",
    "data_edge": "#d5a500",
    "offline": "#dde9ff",
    "offline_edge": "#3f6dd1",
    "online": "#dff5e1",
    "online_edge": "#3aa852",
    "rule": "#ffe1d6",
    "rule_edge": "#d96323",
    "out": "#ececec",
    "out_edge": "#666666",
}


def block(ax, x, y, w, h, title, lines, fill, edge, *, title_size=11.5, line_size=9.5):
    """Рисует блок и автоматически распределяет подписи по высоте,
    чтобы они не вылезали за границы.
    """

    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    title_y = y + h - 0.38
    ax.text(x + w / 2, title_y, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#222")

    if not lines:
        return
    top = title_y - 0.55
    bottom = y + 0.35
    if len(lines) == 1:
        ys = [(top + bottom) / 2]
    else:
        step = (top - bottom) / (len(lines) - 1)
        ys = [top - i * step for i in range(len(lines))]
    for ln, ly in zip(lines, ys):
        ax.text(x + 0.20, ly, "• " + ln,
                ha="left", va="center", fontsize=line_size, color="#222")


def arrow(ax, x1, y1, x2, y2, *, color="#444", width=1.4, style="-|>"):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=width),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.4, 9.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.2)
    ax.set_axis_off()

    ax.text(8, 9.85, "Архитектура рекомендательной системы",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#1a1a1a")

    # ── ROW 1: данные ─────────────────────────────────────────────────────
    block(ax, 0.3, 7.4, 4.6, 1.85,
          "Данные",
          [
              "Каталог статей: ~102 тыс., метаданные",
              "Логи карусели: ~22.7 млн показов и кликов",
              "Тексты статей: title + description",
          ],
          COLORS["data"], COLORS["data_edge"])

    block(ax, 5.5, 7.4, 4.6, 1.85,
          "Офлайн-препроцессинг",
          [
              "Эмбеддинги текста (USER-bge-m3, dim=1024)",
              "Coview-индекс  p(b | a)  из сессий",
              "Обучение LTR + оценка propensity",
          ],
          COLORS["offline"], COLORS["offline_edge"])

    block(ax, 10.7, 7.4, 5.0, 1.85,
          "Хранилища",
          [
              "Векторный индекс эмбеддингов",
              "Coview-таблица соседей",
              "Обученная LTR-модель и веса propensity",
          ],
          COLORS["data"], COLORS["data_edge"])

    # ── ROW 2: candidate generation ───────────────────────────────────────
    block(ax, 0.3, 4.7, 7.4, 2.2,
          "Кандидат-генерация (для статьи a)",
          [
              "Similar pool: kNN top-100 по cosine ∪ coview top-50",
              "Explore pool: per-dept top-30 + cross-dept top-20",
              "Итого ~150 уникальных кандидатов на источник (до 200)",
              "Фильтры: дубликаты, blacklist кросс-департаментов",
          ],
          COLORS["offline"], COLORS["offline_edge"])

    block(ax, 8.3, 4.7, 7.4, 2.2,
          "Фичи на пары (a, b)",
          [
              "Семантика: sim(a, b) по эмбеддингам",
              "Кандидат: log_views, like_rate, comment_rate, fresh",
              "Источник: log_views(a), like_rate(a), fresh(a)",
              "Парные: same_author / dept / rubric, |Δage|",
              "Контекст: позиция, источник пула (sim / explore)",
          ],
          COLORS["offline"], COLORS["offline_edge"])

    # ── ROW 3: ranker + rules ─────────────────────────────────────────────
    block(ax, 0.3, 1.95, 7.4, 2.45,
          "LTR-ранкер",
          [
              "LightGBM LambdaRank — основной",
              "CatBoost YetiRank — альтернатива",
              "Веса w = 1 / p̂(click | position) — debias",
              "Time-split 80/20 по дате события",
              "Test: nDCG@6 = 0.74,  MRR@6 = 0.66",
          ],
          COLORS["online"], COLORS["online_edge"])

    block(ax, 8.3, 1.95, 7.4, 2.45,
          "Reranking и продуктовые правила",
          [
              "Анти-дубликаты по заголовку",
              "Лимиты по автору и по рубрике",
              "Запреты кросс-департамент-переходов",
              "Фильтр explore: похожесть и возраст",
              "Интерливинг:  K = 9 similar + 3 explore",
          ],
          COLORS["rule"], COLORS["rule_edge"])

    # ── ROW 4: output ─────────────────────────────────────────────────────
    block(ax, 4.4, 0.15, 7.2, 1.55,
          "Выход — карусель «Что ещё мы писали»",
          [
              "K = 12 уникальных статей",
              "С разнообразием по авторам, рубрикам и свежести",
          ],
          COLORS["out"], COLORS["out_edge"])

    # ── стрелки ───────────────────────────────────────────────────────────
    # данные → препроцессинг
    arrow(ax, 4.9, 8.3, 5.5, 8.3)
    # препроцессинг → артефакты
    arrow(ax, 10.1, 8.3, 10.7, 8.3)
    # артефакты вниз к фичам
    arrow(ax, 13.2, 7.4, 12.0, 6.92)
    # эмбеддинги → cand-gen
    arrow(ax, 7.5, 7.4, 5.5, 6.92)
    # cand-gen → фичи
    arrow(ax, 7.7, 5.8, 8.3, 5.8)
    # фичи → ranker
    arrow(ax, 12.0, 4.7, 10.5, 4.42)
    # cand-gen → ranker (через скоринг)
    arrow(ax, 4.0, 4.7, 4.0, 4.4)
    # ranker → rerank
    arrow(ax, 7.7, 3.2, 8.3, 3.2)
    # rerank → output
    arrow(ax, 9.7, 1.94, 8.3, 1.7)

    # лёгкая нумерация этапов слева (дополнительный гайд)
    for i, (yc, label) in enumerate([
        (8.32, "1"), (5.8, "2"), (3.2, "3"), (0.92, "4"),
    ]):
        ax.add_patch(mpatches.Circle((-0.05, yc), 0.18, color="#1a1a1a", zorder=5))
        ax.text(-0.05, yc, label, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10, zorder=6)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
    print("saved:", OUT_PNG, OUT_SVG)


if __name__ == "__main__":
    main()
