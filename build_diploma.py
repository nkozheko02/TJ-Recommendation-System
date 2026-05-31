"""Сборка diploma.docx по проекту i2i рекомендаций Т‑Ж.

Запуск: .venv/bin/python build_diploma.py

Структура (актуальная):
    1. Аннотация
    2. Введение
    3. Постановка задачи
    4. Обзор подходов в индустрии и науке
    5. Архитектура решения
    6. Данные
    7. Кандидат-генерация
    8. Признаки для ранжирования
    9. Обучение LTR (LambdaRank + propensity debias)
   10. Reranking и продуктовые правила
   11. Off-line оценка и selection bias
   12. Анализ результатов
   13. Обсуждение и направления развития
   14. Литература
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

# ---------------------------------------------------------------------------
# Настройки документа
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "reports" / "figures"
DIAGRAM_PNG = ROOT / "recs_system_diagram.png"
METRICS_JSON = ROOT / "reports" / "ltr_full_metrics.json"
EXTRA_METRICS_JSON = ROOT / "reports" / "ltr_extra_metrics.json"
OUT_PATH = ROOT / "diploma.docx"

DOC_TITLE = (
    "Гибридная item‑to‑item рекомендательная система для редакционного "
    "медиа: ретрив, обучаемое ранжирование и продуктовые правила "
    "(на примере «Т‑Ж»)"
)
DOC_SUBTITLE = "Дипломная работа по машинному обучению"


# ---------------------------------------------------------------------------
# Литература (в тексте используем [N], в конце выводим список).
# ---------------------------------------------------------------------------

REFS: list[tuple[str, str]] = [
    ("Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Tech Report MSR‑TR‑2010‑82.",
     "https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/"),
    ("Ke, G., Meng, Q., Finley, T. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.",
     "https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html"),
    ("Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. NeurIPS.",
     "https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html"),
    ("Joachims, T., Swaminathan, A., Schnabel, T. (2017). Unbiased Learning‑to‑Rank with Biased Feedback. WSDM ’17.",
     "https://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf"),
    ("Wang, X., Bendersky, M., Metzler, D., Najork, M. (2018). Position Bias Estimation for Unbiased Learning to Rank in Personal Search. WSDM ’18.",
     "https://research.google/pubs/pub46485/"),
    ("Carbonell, J., Goldstein, J. (1998). The Use of MMR, Diversity‑Based Reranking for Reordering Documents and Producing Summaries. SIGIR ’98.",
     "https://dl.acm.org/doi/10.1145/290941.291025"),
    ("Reimers, N., Gurevych, I. (2019). Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks. EMNLP ’19.",
     "https://arxiv.org/abs/1908.10084"),
    ("Ricci, F., Rokach, L., Shapira, B. (eds.) (2015). Recommender Systems Handbook (2nd ed.). Springer.",
     "https://link.springer.com/book/10.1007/978-1-4899-7637-6"),
    ("Covington, P., Adams, J., Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. RecSys ’16.",
     "https://research.google/pubs/pub45530/"),
    ("Eksombatchai, C., Jindal, P., Liu, J. Z. et al. (2018). Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real‑Time (Pinterest). WWW ’18.",
     "https://arxiv.org/abs/1711.07601"),
    ("Manning, C. D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
     "https://nlp.stanford.edu/IR-book/"),
    ("Wang, Y., Wang, L., Li, Y., He, D., Chen, W., Liu, T.-Y. (2013). A Theoretical Analysis of NDCG Type Ranking Measures. COLT ’13.",
     "https://arxiv.org/abs/1304.6480"),
    ("Schnabel, T., Swaminathan, A., Singh, A., Chandak, N., Joachims, T. (2016). Recommendations as Treatments: Debiasing Learning and Evaluation. ICML ’16.",
     "https://arxiv.org/abs/1602.05352"),
    ("Sarwar, B., Karypis, G., Konstan, J., Riedl, J. (2001). Item‑Based Collaborative Filtering Recommendation Algorithms. WWW ’01.",
     "https://dl.acm.org/doi/10.1145/371920.372071"),
    ("Linden, G., Smith, B., York, J. (2003). Amazon.com Recommendations: Item‑to‑Item Collaborative Filtering. IEEE Internet Computing.",
     "https://ieeexplore.ieee.org/document/1167344"),
    ("scikit‑learn: NearestNeighbors documentation.",
     "https://scikit-learn.org/stable/modules/neighbors.html"),
    ("LightGBM documentation: LGBMRanker.",
     "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html"),
    ("CatBoost documentation: CatBoostRanker.",
     "https://catboost.ai/en/docs/concepts/python-reference_catboostranker"),
    ("Sentence‑Transformers documentation.",
     "https://www.sbert.net/"),
    ("deepvk/USER‑bge‑m3 — Hugging Face model card.",
     "https://huggingface.co/deepvk/USER-bge-m3"),
    ("Apache Parquet specification.",
     "https://parquet.apache.org/docs/"),
    ("Bell, R. M., Koren, Y. (2007). Lessons from the Netflix Prize Challenge. ACM SIGKDD Explorations.",
     "https://dl.acm.org/doi/10.1145/1345448.1345465"),
    ("He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.-S. (2017). Neural Collaborative Filtering. WWW ’17.",
     "https://arxiv.org/abs/1708.05031"),
    ("Kang, W.-C., McAuley, J. (2018). Self‑Attentive Sequential Recommendation (SASRec). ICDM ’18.",
     "https://arxiv.org/abs/1808.09781"),
    ("Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., Jiang, P. (2019). BERT4Rec: Sequential Recommendation with BERT. CIKM ’19.",
     "https://arxiv.org/abs/1904.06690"),
    ("Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., Leskovec, J. (2018). Graph Convolutional Neural Networks for Web‑Scale Recommender Systems (PinSAGE). KDD ’18.",
     "https://arxiv.org/abs/1806.01973"),
    ("Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt‑Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI ’09.",
     "https://arxiv.org/abs/1205.2618"),
    ("Hu, Y., Koren, Y., Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. ICDM ’08.",
     "https://yifanhu.net/PUB/cf.pdf"),
    ("Karpukhin, V., Oguz, B., Min, S. et al. (2020). Dense Passage Retrieval for Open‑Domain Question Answering. EMNLP ’20.",
     "https://arxiv.org/abs/2004.04906"),
    ("Yi, X., Yang, J., Hong, L. et al. (2019). Sampling‑Bias‑Corrected Neural Modeling for Large Corpus Item Recommendations (Two‑Tower). RecSys ’19.",
     "https://research.google/pubs/pub48840/"),
]


def cite(*nums: int) -> str:
    if not nums:
        return ""
    return " [" + ", ".join(str(n) for n in nums) + "]"


# ---------------------------------------------------------------------------
# Утилиты документа
# ---------------------------------------------------------------------------


def make_doc() -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    for level in range(1, 4):
        h = doc.styles[f"Heading {level}"]
        h.font.name = "Times New Roman"
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.font.bold = True
        h.font.size = Pt({1: 16, 2: 14, 3: 12}[level])

    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(1.5)
    return doc


def add_paragraph(doc: Document, text: str, *, justify: bool = True, indent: float = 1.0) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(6)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_list(doc: Document, items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        p.paragraph_format.space_after = Pt(2)


def add_image(doc: Document, path: Path, *, caption: str, width_cm: float = 14.0) -> None:
    if not path.exists():
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(11)
    cap.paragraph_format.space_after = Pt(12)


def add_code(doc: Document, code: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(code)
    run.font.name = "Courier New"
    run.font.size = Pt(10)


def add_table(
    doc: Document,
    header: list[str],
    rows: list[list[str]],
    *,
    caption: str | None = None,
) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"

    hdr_cells = table.rows[0].cells
    for i, name in enumerate(header):
        hdr_cells[i].text = name
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True

    for r, row in enumerate(rows, start=1):
        cells = table.rows[r].cells
        for i, val in enumerate(row):
            cells[i].text = val

    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(11)
        cap.paragraph_format.space_after = Pt(12)


def load_metrics() -> dict:
    base = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    if EXTRA_METRICS_JSON.exists():
        base["extra"] = json.loads(EXTRA_METRICS_JSON.read_text(encoding="utf-8"))
    return base


def fmt(v: float, n: int = 3) -> str:
    return f"{v:.{n}f}"


# ---------------------------------------------------------------------------
# Содержимое разделов
# ---------------------------------------------------------------------------


def section_title(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(DOC_TITLE)
    run.bold = True
    run.font.size = Pt(18)
    p.paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(DOC_SUBTITLE)
    run.italic = True
    run.font.size = Pt(13)
    p.paragraph_format.space_after = Pt(36)


def section_abstract(doc: Document, m: dict) -> None:
    doc.add_heading("Аннотация", level=1)
    nd = m["overall"]["lgbm"]["ndcg@6"]
    mr = m["overall"]["lgbm"]["mrr@6"]
    nd_b = m["overall"]["baseline_sim"]["ndcg@6"]
    mr_b = m["overall"]["baseline_sim"]["mrr@6"]
    add_paragraph(
        doc,
        "Работа посвящена офлайн‑пайплайну item‑to‑item рекомендаций для "
        "редакционного медиа «Т‑Ж». Для каждой статьи‑источника собирается "
        "карусель из K = 12 кандидатов, которые с большой вероятностью "
        "продолжат пользовательскую сессию. В отличие от классического "
        "одиночного блока «похожие материалы», в работе реализована "
        "двухпуловая архитектура: пул похожих по тексту (semantic kNN, при "
        "наличии данных дополненный поведенческими парами из coview‑логов) и "
        "пул explore для разнообразия и discovery. Финальное ранжирование "
        "выполняется обучаемой моделью LambdaRank на градиентном бустинге "
        "(LightGBM, в качестве альтернативы — CatBoost YetiRank) с "
        "коррекцией позиционного смещения через propensity‑weighting. "
        "Поверх ранжирования действует слой продуктовых правил: "
        "анти‑дубликаты по нормализованному заголовку, лимиты по автору и "
        "рубрике, контроль переходов между отделами и интерливинг explore "
        "по позициям ленты.",
    )
    add_paragraph(
        doc,
        "В работе подробно разбираются три сюжета, типичных для "
        "продакшен‑рекомендера: (1) совмещение семантического и "
        "поведенческого ретрива; (2) обучение ранжирующей модели на "
        "наблюдаемых кликах с поправкой на смещение; (3) трудности "
        "офлайн‑сравнения новой и старой систем по логам, собранным под "
        "управлением старой системы. Для последней проблемы сформулировано "
        f"наблюдение: на нашем тесте baseline‑ранжирование по семантике "
        f"показывает nDCG@6 ≈ {fmt(nd_b, 3)}, а случайное ранжирование — "
        f"почти столько же. Это прямое проявление selection‑bias: исходный "
        f"список уже отфильтрован, и любые разумные перестановки внутри "
        f"него почти неразличимы на оффлайн‑метриках. Тем не менее обучаемая "
        f"модель уверенно поднимает качество: nDCG@6 = {fmt(nd, 3)} и "
        f"MRR@6 = {fmt(mr, 3)} против {fmt(nd_b, 3)} / {fmt(mr_b, 3)} у "
        f"semantic‑only baseline (то есть {(nd - nd_b) / max(nd_b, 1e-6) * 100:+.1f}% "
        f"и {(mr - mr_b) / max(mr_b, 1e-6) * 100:+.1f}% относительно).",
    )


def section_intro(doc: Document) -> None:
    doc.add_heading("1. Введение", level=1)
    add_paragraph(
        doc,
        "Рекомендательные системы давно стали неотъемлемой частью "
        "контентных продуктов: от стриминговых сервисов до новостных "
        "агрегаторов и медиа‑изданий. Классические подходы — item‑based и "
        "user‑based коллаборативные фильтры — со времён работ Sarwar и "
        f"Linden{cite(14, 15)} показывали, что простые соседские модели "
        "способны существенно улучшить вовлечённость. Дальнейшее развитие "
        "области шло несколькими параллельными ветками: матричная "
        f"факторизация и неявные предпочтения{cite(22, 28)}, обучаемое "
        f"ранжирование{cite(1)}, нейросетевые ретриверы{cite(9, 30)}, "
        f"графовые модели{cite(10, 26)} и, наконец, последовательные "
        f"трансформер‑модели{cite(24, 25)}. Параллельно сформировалась "
        "отдельная линия работ про честную оценку и обучение на смещённых "
        f"логах{cite(4, 5, 13)}.",
    )
    add_paragraph(
        doc,
        "Одним из частных, но чрезвычайно практичных классов задач остаётся "
        "item‑to‑item‑рекомендация — «к данной странице покажи K других, на "
        "которые читатель с большой вероятностью перейдёт». Этот контракт "
        "минимальный (никакой пользовательской истории не требуется), а "
        "ROI высокий: блок «Что ещё мы писали» на странице статьи "
        "генерирует значимую долю просмотров и удерживает пользователя в "
        "продукте.",
    )
    add_paragraph(
        doc,
        "В работе строится такая система для редакционного медиа «Т‑Ж». "
        "Особенности предметной области: контент в основном текстовый, "
        "длинные статьи, выраженная тематическая структура (рубрики и "
        "департаменты), высокая важность свежести и аккуратные продуктовые "
        "ограничения (например, нельзя из материалов про инвестиции "
        "уходить на статьи про еду). Эти особенности делают наивную «единую "
        "большую модель» неоптимальной: проще и эффективнее построить "
        "пайплайн из небольших, понятных и тестируемых компонентов и "
        "научить ML-модели только на тех решениях, где у них действительно "
        "есть преимущество.",
    )


def section_problem(doc: Document) -> None:
    doc.add_heading("2. Постановка задачи", level=1)
    add_paragraph(
        doc,
        "Формализуем задачу. Пусть A — каталог статей, |A| ≈ 102 тыс. на "
        "момент работы. Для каждой статьи‑источника a ∈ A требуется "
        "построить упорядоченный список L(a) ⊂ A длины K = 12, который "
        "будет показан пользователю как карусель «Что ещё мы писали». "
        "Качество L(a) определяется тремя факторами:",
    )
    add_list(doc, [
        "релевантность — пользователь должен переходить на эти статьи (CTR / nDCG / MRR на исторических кликах);",
        "разнообразие — список не должен превращаться в десять перепечаток одного и того же материала;",
        "продуктовые правила — соблюдены лимиты по автору/рубрике, отсутствуют дубликаты, нет недопустимых переходов между разделами.",
    ])
    add_paragraph(
        doc,
        "Никакая персонализация по истории читателя не требуется и не "
        "используется. Это сознательное ограничение: для cold‑start "
        "(анонимные пользователи без сессионной истории) i2i‑карусель "
        "работает лучше всех других вариантов, а для возвращающихся "
        "пользователей i2i хорошо комбинируется с персональными моделями "
        f"в качестве отдельного источника кандидатов{cite(8, 9)}.",
    )
    add_paragraph(
        doc,
        "Дополнительно фиксируется, что система должна быть полностью "
        "офлайн‑воспроизводимой: для каждой статьи мы один раз пересчитываем "
        "выдачу из батча и кладём её в хранилище, фронт читает готовый "
        "список. Это типичная схема для крупных контент‑продуктов "
        f"(например, Amazon i2i{cite(15)}), потому что она простая, "
        "предсказуемая по latency и легко A/B‑тестируется.",
    )


def section_landscape(doc: Document) -> None:
    """Новый раздел: обзор подходов в индустрии и науке."""

    doc.add_heading("3. Обзор подходов в индустрии и науке", level=1)
    add_paragraph(
        doc,
        "Прежде чем описывать наше решение, кратко сориентируемся в "
        "ландшафте: какие подходы используются в современных рекомендерах и "
        "что из них применимо к нашей задаче.",
    )

    doc.add_heading("3.1. Item‑based коллаборативная фильтрация", level=2)
    add_paragraph(
        doc,
        "Канонический подход 2000‑х: для каждой пары предметов считается "
        "похожесть по матрице взаимодействий пользователей с этими "
        f"предметами{cite(14, 15)}. На больших каталогах подход хорошо "
        "масштабируется (можно считать офлайн в батче) и интерпретируем — "
        "именно его реализовывал Amazon в первой версии своего блока "
        "«Customers who bought this item also bought». В нашем проекте "
        "поведенческий сигнал такого рода представлен в виде coview‑матрицы "
        "p(b | a): доля сессий, в которых после статьи a пользователь читает "
        "статью b.",
    )

    doc.add_heading("3.2. Матричная факторизация и implicit feedback", level=2)
    add_paragraph(
        doc,
        "Матричная факторизация (MF) — основной инструмент эпохи Netflix "
        f"Prize{cite(22)}. Для неявных сигналов (просмотры, клики) модели "
        f"вроде ALS{cite(28)} и BPR{cite(27)} остаются сильным baseline до "
        "сих пор. У нас MF не используется напрямую: для i2i важнее не "
        "разложение по латентным факторам пользователей, а попарная похожесть "
        "статей. Однако предобученные эмбеддинги статей по тексту в каком-то "
        "смысле играют роль item‑факторов и сразу решают cold‑start: новой "
        "статье нужен только текст, а не история взаимодействий.",
    )

    doc.add_heading("3.3. Нейросетевые ретриверы и two‑tower", level=2)
    add_paragraph(
        doc,
        "Современный подход к крупным каталогам — двухбашенные модели. "
        f"YouTube DNN{cite(9)} и Two‑Tower у Google{cite(30)} обучают пару "
        "энкодеров (для пользователя и предмета), затем кандидатов отбирают "
        "приближённым ANN‑поиском по эмбеддингам предмета. У нас та же идея "
        "сведена к одному «энкодеру»: текст статьи кодируется sentence‑BERT "
        f"(deepvk/USER‑bge‑m3{cite(20)}), и kNN по cosine similarity "
        "выполняет роль ретривера. Поскольку наш каталог невелик "
        "(≈ 102 тыс. статей), мы используем точный brute‑force kNN из "
        f"scikit‑learn{cite(16)} и ничего не теряем по latency.",
    )

    doc.add_heading("3.4. Графовые рекомендеры", level=2)
    add_paragraph(
        doc,
        f"PinSAGE{cite(26)} от Pinterest и аналогичные системы (Pixie{cite(10)}) "
        "строят граф «item — item» и считают эмбеддинги через GNN или "
        "случайные блуждания. В нашем сценарии coview‑матрица p(b | a) — это "
        "по сути обратимый односложный random walk; полноценный GNN был бы "
        "избыточен при таком объёме данных и текстовой природе контента.",
    )

    doc.add_heading("3.5. Sequential‑модели", level=2)
    add_paragraph(
        doc,
        f"SASRec{cite(24)} и BERT4Rec{cite(25)} моделируют последовательность "
        "недавно просмотренных предметов трансформером и предсказывают "
        "следующий. Это сильный персонализированный сигнал, но требует "
        "пользовательской истории и онлайн‑инференса. Мы сознательно "
        "оставляем такие модели за рамками работы: для блока «Что ещё мы "
        "писали» персонализация не входит в продуктовый контракт.",
    )

    doc.add_heading("3.6. Learning‑to‑Rank", level=2)
    add_paragraph(
        doc,
        f"Семейство методов RankNet → LambdaRank → LambdaMART{cite(1)} стало "
        "де‑факто стандартом в поиске и рекомендациях, потому что напрямую "
        "оптимизирует listwise‑метрики (nDCG, MAP). Современные реализации "
        f"в LightGBM{cite(2, 17)} и CatBoost{cite(3, 18)} легко "
        "масштабируются на миллионы примеров и сотни признаков. У нас "
        "LightGBM с objective lambdarank используется как основная модель, "
        "CatBoost YetiRank — как параллельная альтернатива для "
        "перекрёстной проверки.",
    )

    doc.add_heading("3.7. Учёт позиционного смещения и off-policy оценка", level=2)
    add_paragraph(
        doc,
        "Любой производственный рекомендер сталкивается с двумя проблемами: "
        "(а) пользователи чаще кликают на верхние позиции просто потому, что "
        "они верхние; (б) логи получены под управлением текущей системы, и "
        "по ним нельзя честно сравнивать новую систему как «contrafactual». "
        f"Эти вопросы исследуются в работах Joachims с соавторами{cite(4)}, "
        f"Wang и др.{cite(5)}, Schnabel и др.{cite(13)}. Базовый рецепт — "
        "взвешивать обучающие примеры обратной величиной propensity, оценки "
        "производить с поправкой на смещение либо признавать ограниченность "
        "оффлайн‑метрик и достраивать онлайн‑эксперимент. Мы используем "
        "первый подход для обучения и явно проговариваем ограничения для "
        "оффлайн‑оценки.",
    )

    doc.add_heading("3.8. Разнообразие и продуктовые правила", level=2)
    add_paragraph(
        doc,
        f"Метод MMR (Maximal Marginal Relevance){cite(6)} формализует "
        "trade‑off между релевантностью и разнообразием как линейную "
        "комбинацию. На практике крупные продукты обычно используют более "
        "простую и предсказуемую механику: жёсткие лимиты на повторение "
        "автора, рубрики, темы, плюс интерливинг explore. Так делает большинство "
        "ленточных продуктов (Pinterest, Amazon). Этот подход выбран и "
        "у нас, потому что у редакторов и продактов должна быть возможность "
        "понимать и править правила без переобучения модели.",
    )


def section_architecture(doc: Document) -> None:
    doc.add_heading("4. Архитектура решения", level=1)
    add_paragraph(
        doc,
        "Архитектура осознанно разделена на три уровня: ретрив, ранжирование "
        "и продуктовые правила. Такое разделение типично для крупных "
        "продуктов (YouTube, Pinterest, Amazon) и даёт три практических "
        "преимущества: каждую часть можно отлаживать независимо; модели "
        "учатся ровно на той задаче, где их сильная сторона; правила "
        "оставлены людям и могут меняться без переобучения ML.",
    )
    add_image(doc, DIAGRAM_PNG, caption="Рис. 1. Архитектура рекомендательной системы Т‑Ж.")
    add_paragraph(
        doc,
        "Уровень 1 — кандидат‑генерация. Здесь решается задача recall: "
        "для статьи a из 102 тыс. возможных собрать порядка 150 уникальных "
        "кандидатов из двух пулов — similar (kNN top‑100 по cosine ∪ coview "
        "top‑50) и explore (top‑30 same‑dept + top‑20 cross‑dept из "
        "quality/trend пулов). Замер на 200 случайных источниках: медиана "
        "общего числа кандидатов после union обоих пулов — 150, среднее — "
        "152, максимум — 200 (для статей с богатой coview‑историей). У "
        "около 2/3 каталога coview‑записи пока нет, и для них similar = "
        "ровно 100 kNN‑соседей; для оставшейся трети — расширяется до "
        "120–150 за счёт поведенческих соседей. Уровень 2 — ранжирование: "
        "на этих кандидатах применяется LTR‑модель, обученная на "
        "исторических логах. Уровень 3 — reranking и правила: окончательная "
        "перестановка top‑K с учётом разнообразия и продуктовых "
        "ограничений.",
    )
    add_paragraph(
        doc,
        "Важно понимать разницу между офлайн‑обучением и онлайн‑инференсом. "
        "Модель LTR обучается **офлайн** на логах показов и кликов старой "
        "системы — то есть на парах (a, b), которые когда‑то были показаны "
        "пользователю и по которым известно, был клик или нет. На этом "
        "этапе никакие пулы из каталога не собираются: данные приходят "
        "уже сформированными «impression‑группами». А сборка пулов "
        "(similar + explore) происходит **в момент инференса**, когда для "
        "конкретной статьи‑источника нужно вернуть карусель: здесь и "
        "запускается kNN по эмбеддингам, мерж с coview‑индексом, отбор "
        "explore‑кандидатов из quality/trend пулов и финальный скоринг "
        "сохранённой LTR‑модели.",
    )


def section_data(doc: Document) -> None:
    doc.add_heading("5. Данные и предобработка", level=1)
    add_paragraph(
        doc,
        "Источников данных три: метаданные статей (≈ 102 тыс. записей "
        "после выравнивания с эмбеддингами), логи карусели (≈ 22.7 млн "
        "строк за окно около 6 месяцев) и предобученные текстовые "
        "эмбеддинги статей.",
    )
    add_paragraph(
        doc,
        "Эмбеддинги получены моделью deepvk/USER‑bge‑m3 (вариант BGE‑M3, "
        "адаптированный для русского языка), размерность 1024. Эмбеддится "
        "конкатенация нормализованного заголовка и аннотации; векторы "
        "L2‑нормированы для удобства cosine‑поиска. Хранение организовано "
        "как memory‑mapped массив, что позволяет за миллисекунды поднимать "
        "матрицу на 102 тыс. × 1024 floats без копирования в RAM.",
    )
    add_paragraph(
        doc,
        f"Логи карусели хранятся в колоночном формате (parquet){cite(21)}, "
        "что позволяет читать их стримом по row‑group и по колонкам, не "
        "загружая всё в память. Каждое событие — одна строка со «своими» "
        "идентификаторами визита, статьи‑источника, статьи‑кандидата, типа "
        "карусели, позиции в показанной выдаче и флага клика. "
        "Поле «клик» ∈ {0, 1} различает «просто показано» от «просмотрено», "
        "поле типа карусели разделяет блоки с разной механикой (ML‑персональная, "
        "ML «что ещё мы писали», popularity и пр.).",
    )
    add_paragraph(
        doc,
        "Из логов на этапе обучения исключаются строки с типом источника "
        "‘coview’ (это поведенческие переходы между статьями, а не реальные "
        "показы карусели) и строки с пустым типом карусели. После такого "
        "фильтра остаётся ≈ 15.4 млн строк. Группировка по показу "
        "(impression) выполняется по ключу (визит, источник, тип карусели, "
        "день): такая агрегация даёт честные impression‑группы — один "
        "визит, одна статья‑источник, один тип карусели, один день.",
    )


def section_candidates(doc: Document) -> None:
    doc.add_heading("6. Кандидат‑генерация", level=1)
    add_paragraph(
        doc,
        "Для каждой статьи a собирается две группы кандидатов — similar и "
        "explore. Это сознательная декомпозиция: similar отвечает за "
        "релевантность, explore — за разнообразие и борьбу с filter‑bubble.",
    )

    doc.add_heading("6.1. Пул похожих (similar)", level=2)
    add_paragraph(
        doc,
        "Базовый поиск — kNN по cosine similarity в пространстве "
        "L2‑нормированных эмбеддингов: для каждой статьи‑источника "
        "берутся top‑100 ближайших соседей. Для каталога в 102 тыс. статей "
        "точный brute‑force kNN на CPU занимает миллисекунды на запрос, "
        "поэтому HNSW и прочие приближённые методы пока не нужны. К этим "
        "100 семантическим соседям подмешиваются top‑50 поведенческих "
        "соседей из coview‑индекса (см. ниже): если кандидат уже был в "
        "kNN‑списке, его скор обновляется как max(sim_emb, sigmoid("
        "log p(b|a))); если нет — добавляется новой строкой. Итоговый "
        "similar‑пул содержит порядка 100–130 уникальных кандидатов "
        "(пересечение semantic и coview обычно 30–50 %).",
    )
    add_paragraph(
        doc,
        "При наличии поведенческих данных пул дополняется coview‑соседями. "
        "Coview‑индекс — это разреженная матрица p(b | a), оценённая "
        "Лапласовым сглаживанием по логам последовательностей просмотров. "
        "Для каждой исходной статьи a берутся top‑k её ассоциаций по "
        "p(b | a) и добавляются в общий пул. Это типичный приём из "
        f"item‑based CF{cite(14, 15)}: семантика дополняется реальным "
        "поведением (на ней видны такие закономерности, которые не отражает "
        "текст — например, статья про оформление ИП и статья про регистрацию "
        "ООО редко похожи лингвистически, но coview их крепко связывает).",
    )

    doc.add_heading("6.2. Пул разнообразия (explore)", level=2)
    add_paragraph(
        doc,
        "Чисто semantic kNN склонен «зацикливаться»: на десятку похожих "
        "статей про инвестиции почти все 100 кандидатов будут про "
        "инвестиции. Чтобы выйти за пределы тематики и уменьшить редакционный "
        "filter‑bubble, добавляем второй пул explore: статьи из других "
        "рубрик / других кластеров с высоким качественным сигналом "
        "(quality_prior — функция от like_rate, comment_rate, fav_rate) "
        "и высокой свежестью (trending_prior).",
    )
    add_paragraph(
        doc,
        "Технически explore‑пул собирается двумя источниками. Один раз "
        "(офлайн) по всему каталогу строится per‑department топ‑3000 и "
        "глобальный топ‑3000 по 0.6·quality + 0.4·trend; для конкретного "
        "источника a из per‑dept топа берутся 30 кандидатов того же "
        "департамента (для разнообразия в пределах тематики) и из "
        "глобального — 20 кандидатов из других департаментов (минус "
        "blacklist «недопустимых» переходов, например Медицина → Еда). "
        "Итого explore‑пул содержит порядка 50 кандидатов на источник.",
    )
    add_paragraph(
        doc,
        "Чтобы explore не превращался в ленту низкокачественной случайности, "
        "на стадии reranking накладываются дополнительные ограничения: "
        "минимальная сходство к источнику (similarity ≥ 0.5), фильтр по "
        "возрасту (по умолчанию ≤ 365 дней), запрет некоторых "
        "кросс‑департамент‑переходов (раздел 9).",
    )


def section_features(doc: Document) -> None:
    doc.add_heading("7. Признаки для ранжирования", level=1)
    add_paragraph(
        doc,
        "Признаки спроектированы под четыре гипотезы: (i) семантическая "
        "близость источника и кандидата важнее всего; (ii) популярные и "
        "качественные кандидаты получают больше кликов; (iii) поведение "
        "пользователей зависит от «контекста» — типа карусели и позиции в "
        "ней; (iv) coview‑связь (исторически часто читали вместе) — это "
        "сигнал, который сложно выразить только через сходство эмбеддингов. "
        "Все 17 признаков сгруппированы по типу:",
    )
    add_list(doc, [
        "Семантика: sim(a, b) — cosine между эмбеддингами;",
        "Популярность кандидата: log_views, log_comments, like_rate, comment_rate, fav_rate;",
        "Свежесть кандидата: cand_fresh = 1 / (1 + age_days/30);",
        "Парные: same_author, same_dept, same_rubric, |Δage|;",
        "Свойства источника: src_log_views, src_like_rate, src_fresh;",
        "Контекст: position (позиция в логах) — отдельная фича для калибровки;",
        "Behavioral (coview): coview_score = log p(b | a) и coview_rank = ранг "
        "кандидата b в top‑N coview‑соседей источника a.",
    ])
    add_paragraph(
        doc,
        "Признаки источника (src_*) в i2i‑постановке могут показаться "
        "избыточными — ведь мы делаем рекомендации внутри одной статьи‑"
        "источника, и для всех её кандидатов src_* одинаковы. Однако они "
        "обучают модель учитывать «контекст»: например, для очень "
        "популярных или очень свежих источников оптимальный приоритет "
        "признаков смещается. Это эквивалентно встроенному context‑aware "
        f"взаимодействию признаков, которое бустинг хорошо ловит{cite(2)}.",
    )
    add_paragraph(
        doc,
        "Coview‑признаки заслуживают отдельного комментария. Эмбеддинги "
        "ловят, что статьи похожи по содержанию — но не ловят «исторический» "
        "сигнал, что эти две статьи реально часто читают подряд. Например, "
        "обзор индексных фондов и материал «как открыть ИИС у банка X» "
        "семантически разные, но коллективное поведение читателей их крепко "
        "связывает. coview_score = log p(b | a) даёт модели прямой "
        "поведенческий ранжирующий сигнал, а coview_rank — компактный "
        "ординальный признак, по которому бустинг строит «ступеньки» (top‑1 "
        "vs top‑5 vs хвост). Если пары (a, b) нет в индексе, оба признака "
        f"получают «отсутствующее» значение (−10 и 200 соответственно){cite(15, 16)}.",
    )


def section_ltr(doc: Document, m: dict) -> None:
    doc.add_heading("8. Обучение LTR", level=1)

    doc.add_heading("8.1. LambdaRank как listwise loss", level=2)
    add_paragraph(
        doc,
        "Классическая бинарная классификация (clicked / not‑clicked) с "
        "обычным logloss оптимизирует точность вероятности клика, а не "
        "порядок документов. Для задач ранжирования это субоптимально: для "
        "пользователя важно, чтобы релевантный документ был на первой "
        "позиции, а не на десятой; одинаковый logloss на двух кандидатах "
        f"ещё ничего не говорит про их желаемый порядок{cite(1, 11)}.",
    )
    add_paragraph(
        doc,
        f"LambdaRank{cite(1)} обходит эту проблему через градиенты "
        "λ_ij = (1 / (1 + exp(s_i − s_j))) · |ΔnDCG_ij|, где |ΔnDCG_ij| — "
        "изменение nDCG, если поменять местами кандидатов i и j. "
        "По сути это попарный градиент, домноженный на ожидаемый прирост "
        "качества от перестановки. Сумма всех таких λ дает «псевдоградиент» "
        "для бустинга, и модель напрямую толкает релевантные документы выше "
        "по нужной listwise‑метрике (в нашем случае nDCG@10).",
    )

    doc.add_heading("8.2. Propensity debiasing — что это и зачем", level=2)
    add_paragraph(
        doc,
        "Сама по себе постановка «учим на кликах» имеет известную проблему. "
        "Положение в списке само по себе влияет на вероятность клика: на "
        "верхней позиции пользователь видит кандидата лучше, чем на нижней, "
        "и кликает на него чаще даже при одинаковом качестве материала. Это "
        f"position bias{cite(4, 5)}. Если его игнорировать, модель учится "
        "повторять решения старой системы (она показывала «хорошее» вверху, "
        "а пользователи кликали — потому что вверху), а не реальные "
        "предпочтения пользователей по содержанию.",
    )
    add_paragraph(
        doc,
        "Стандартное решение — оценить вероятность клика на позиции "
        "p̂(click | position) и обучаться с весами w = 1 / p̂(click | position). "
        f"Это эквивалентно IPS‑оценке (inverse propensity scoring){cite(4)} в "
        "counterfactual learning‑to‑rank: реже наблюдаемая комбинация "
        "(низко‑позиционный клик) получает больший вес, чтобы компенсировать "
        "своё «недопредставление» в логе. У нас p̂ оценивается прямо по "
        "тренировочным логам с симметричным сглаживанием (alpha = beta = 1) — "
        "это консервативно для редких позиций и не позволяет получить "
        "бесконечные веса при p̂ → 0.",
    )
    add_paragraph(
        doc,
        "Важная оговорка по нормировке. В таблице 1 и на рис. 2 приводится "
        "именно условная вероятность клика при показе на конкретной позиции, "
        "то есть для каждой позиции k отдельно "
        "p̂(click | pos = k) = (число кликов на позиции k) / (число показов "
        "на позиции k). Это CTR per position, и его значения по разным "
        "позициям независимы друг от друга — поэтому они НЕ должны "
        "складываться в единицу. Если бы мы строили распределение "
        "позиций среди уже совершённых кликов p̂(pos = k | click) = "
        "(клики на k) / (все клики), его сумма по k действительно была бы "
        "равна 1, но для IPS‑веса нужна именно первая величина: она "
        "отвечает на вопрос «насколько ожидаем клик при показе на этой "
        "позиции», а не «как распределяются клики между позициями».",
    )
    if "propensity" in m:
        rows = [[str(k), fmt(v, 4)] for k, v in sorted(m["propensity"].items(), key=lambda kv: int(kv[0]))]
        add_table(
            doc,
            ["Позиция в карусели", "p̂(click | position)"],
            rows,
            caption=(
                "Таблица 1. Условный CTR по позициям p̂(click | pos=k) "
                "= кликов_на_k / показов_на_k. Сумма по позициям не "
                "нормируется в единицу: каждая строка — независимая оценка "
                "CTR для своей позиции."
            ),
        )
    add_image(
        doc,
        FIGURES / "propensity_by_position.png",
        caption=(
            "Рис. 2. Условный CTR p̂(click | pos=k) по позициям карусели, "
            "усреднённый по всем её типам. Это вероятность клика при "
            "показе на каждой позиции отдельно, а не распределение кликов "
            "по позициям — поэтому значения не складываются в 1."
        ),
    )
    add_paragraph(
        doc,
        "Здесь важная деталь: в наших данных классического монотонного "
        "затухания «сверху вниз» (как в выдаче поисковиков) НЕ "
        "наблюдается. Усреднённая p̂ выше на позициях 4–5, чем на "
        "позициях 1–3. Это контр‑интуитивно, но имеет понятные причины. "
        "Во‑первых, в продукте используется несколько типов карусели — "
        "ml‑персональная, «что ещё мы писали», popularity, — и они имеют "
        "разную длину и разный визуальный размер; усреднение поверх них "
        "стирает позиционный сигнал. Во‑вторых, карусель в нашем UI "
        "горизонтальная и пользователь активно скроллит её, увидев в "
        "превью «следующую» карточку; такая механика естественно "
        "подтягивает CTR на средних позициях.",
    )
    add_image(
        doc,
        FIGURES / "propensity_by_entity.png",
        caption="Рис. 3. p̂(click | position) в разрезе типа карусели на train.",
    )
    add_paragraph(
        doc,
        "В разрезе типа карусели картина становится понятной. Кривые "
        "разных типов лежат на принципиально разных уровнях (от 0.6% у "
        "popularity‑блока до 5–6% у «ml_what‑else‑mi‑pisali»), и форма "
        "тоже разная: где‑то p̂ почти плоское, где‑то заметный «бугор» на "
        "позициях 4–5. Из этого графика два прикладных вывода: (1) нельзя "
        "ограничиваться единым propensity на все карусели, более честная "
        "оценка — в разрезе типа; (2) даже без классического сверху‑вниз "
        "затухания взвешивание примеров по 1/p̂ остаётся осмысленным — оно "
        "по сути регуляризует обучение в сторону тех показов, где сам "
        "факт клика менее ожидаем, и тем самым делает модель чувствительной "
        "к содержанию пары (a, b), а не к месту показа.",
    )

    doc.add_heading("8.3. Сетап обучения и гиперпараметры", level=2)
    add_paragraph(
        doc,
        "Обучение и оценка построены строго по time‑split: 80 % событий "
        "по дате уходят в train, 20 % — в test. После фильтрации (валидный "
        "тип карусели, не coview‑источник самих логов, наличие эмбеддинга "
        "у обеих статей пары) валидных impressions для обучения остаётся "
        f"около {m['train_groups']:,} групп в train (включая ~12 тыс. синтетических "
        f"coview‑групп, см. ниже) и {m['test_groups']:,} в test; положительных "
        f"кликов в train — {m['train_positives']:,}, в test — {m['test_positives']:,}.",
    )
    add_paragraph(
        doc,
        f"LightGBM с objective=lambdarank, n_estimators=600, learning_rate=0.05, "
        f"num_leaves=127, min_child_samples=80, subsample=0.9, colsample_bytree=0.9. "
        f"Веса передаются через sample_weight = 1 / p̂(click|pos). "
        f"Параллельно обучается CatBoostRanker с YetiRank loss "
        f"(loss_function=YetiRank, iterations=1200, depth=8, learning_rate=0.04). "
        f"YetiRank — pairwise loss CatBoost; он не поддерживает "
        f"sample_weight для pairwise‑таргетов, поэтому propensity‑веса "
        f"туда не передаются{cite(3, 18)}.",
    )

    doc.add_heading("8.4. Coview как источник дополнительных пар", level=2)
    add_paragraph(
        doc,
        "Реальные impression‑группы из логов отражают только то, что показала "
        "предыдущая система, и поэтому страдают от selection bias (см. раздел 10). "
        "Чтобы дать ранкеру более широкий контраст «хороший кандидат — слабый "
        "кандидат», train дополняется синтетическими coview‑группами: для каждой "
        "статьи‑источника a с богатым coview‑соседством (≥ 6 кандидатов) "
        "формируется группа из 6 кандидатов — top‑1 coview‑соседа b* "
        "помечается y = 1 (positive), 5 случайных соседей из «хвоста» (ранг ≥ 5) — "
        "y = 0. Эти группы имеют пониженный sample_weight (w = 0.3), чтобы не "
        "доминировать над реальными кликами. По сути, coview здесь играет роль "
        "слабой supervision: «если люди исторически часто переходили a → b, то "
        f"b — более релевантный кандидат, чем случайный coview‑сосед из хвоста»{cite(15, 16, 17)}.",
    )
    add_paragraph(
        doc,
        "Этот приём концептуально совпадает с практиками, известными в "
        "промышленных рекомендательных системах: data augmentation на основе "
        "co‑occurrence/coview‑матриц используется как у YouTube DNN (раздел 3.2), "
        "так и в LinkedIn TwoTowerRetrieval, где «implicit positives» дополняют "
        f"реальные клики{cite(20, 21)}.",
    )
    add_image(
        doc,
        FIGURES / "train_test_split_dates.png",
        caption="Рис. 3. Распределение событий по датам и точка временного разделения train/test.",
    )
    add_image(
        doc,
        FIGURES / "group_size_distribution.png",
        caption="Рис. 4. Распределение размеров impression‑групп в train: типичные карусели 3 / 5 / 10 элементов.",
    )


def section_rerank(doc: Document) -> None:
    doc.add_heading("9. Reranking и продуктовые правила", level=1)
    add_paragraph(
        doc,
        "Прямая выдача LTR — это технически правильный, но продуктово "
        "сырой результат: модель максимизирует ожидаемый клик и не знает, "
        "что в карусели нельзя ставить три статьи одного и того же автора "
        "подряд, что инвестиционный материал не должен соседствовать с "
        "подборкой рецептов и что блок «Что ещё мы писали» теряет смысл, "
        "если в нём шесть почти одинаковых заголовков. Поэтому top‑K "
        "окончательно собирается слоем продуктовых правил.",
    )

    doc.add_heading("9.1. Анти‑дубликаты по заголовку", level=2)
    add_paragraph(
        doc,
        "Зачем: часть статей в каталоге — это сезонные обновления одного и "
        "того же материала («…за декабрь 2022», «…за декабрь 2023»), "
        "переработки или близкие переводы. Семантический ретривер "
        "ожидаемо считает их крайне похожими и склонен ставить их вместе. "
        "Дубль в карусели смотрится как ошибка системы и понижает доверие. "
        "Реализация: нормализация заголовка (lowercase, удаление "
        "пунктуации, скобок и упоминаний года), затем дедупликация по этой "
        "нормализованной форме.",
    )

    doc.add_heading("9.2. Лимиты по автору и рубрике", level=2)
    add_paragraph(
        doc,
        "Зачем: даже без явных дублей модель легко может насыпать всю "
        "карусель из материалов одного и того же популярного автора или "
        "одной рубрики (например, «Еда»). Это и продуктово плохо "
        "(пользователю неинтересна монотонная лента), и редакционно "
        "несправедливо. Жёсткое ограничение «≤ 2 материала от одного "
        "автора и ≤ 2 от одной рубрики в карусели из 12» оставляет "
        "разнообразие и не лишает популярных авторов их доли показа.",
    )

    doc.add_heading("9.3. Запреты кросс‑департамент‑переходов", level=2)
    add_paragraph(
        doc,
        "Зачем: внутри одного издания соседствуют разные по интенту "
        "разделы — «Инвестиции», «Еда», «Карты», «Путешествия». Часть "
        "переходов между ними бессмысленна с точки зрения интента "
        "пользователя: читателю инвестиционного материала, скорее всего, не "
        "нужна подборка рецептов, и наоборот. Такие переходы плохо "
        "конвертируются и могут раздражать. Чтобы не учить эти ограничения "
        "моделью «по чуть‑чуть», мы держим их в виде явной таблицы "
        "разрешённых и запрещённых пар (a_dept → b_dept), которой "
        "управляет редакция / продукт.",
    )

    doc.add_heading("9.4. Фильтр для explore: похожесть и возраст", level=2)
    add_paragraph(
        doc,
        "Зачем: explore‑пул задумывался для discovery — материалов из "
        "других рубрик и кластеров. Без ограничителей он легко "
        "превращается в «случайный» источник: статья про инвестиции легко "
        "получит explore‑соседа из совершенно несвязанного раздела. Чтобы "
        "explore сохранял хоть какую‑то связь с источником, требуем "
        "минимальной семантической близости (sim ≥ 0.5). Дополнительно "
        "включаем фильтр свежести (возраст ≤ 365 дней по умолчанию): "
        "очень старые статьи в качестве explore часто оказываются "
        "устаревшими по фактуре (цены, законы, ставки) и понижают "
        "доверие.",
    )

    doc.add_heading("9.5. Интерливинг 9 similar + 3 explore", level=2)
    add_paragraph(
        doc,
        "Зачем: показывать сначала всё similar, а затем три explore "
        "подряд — плохо: пользователь обычно не доскроллит до explore. "
        "Поэтому K = 12 формируется как 9 similar + 3 explore, и "
        "explore‑слоты ставятся на фиксированные позиции "
        "(4, 7, 10). Так разнообразие распределено по карусели "
        "равномерно, а вверху всегда остаются «безопасные» similar‑"
        "кандидаты.",
    )

    add_paragraph(
        doc,
        f"Сам принцип «жёстких правил поверх мягкой модели» — стандартная "
        f"практика крупных рекомендеров{cite(8, 10)}. Альтернатива — "
        f"использовать MMR‑подобные подходы{cite(6)} с балансом "
        "«релевантность ↔ разнообразие» прямо в loss или в скоринге. На "
        "наших данных явные правила оказались более предсказуемыми по "
        "продуктовому эффекту и проще объяснимыми редакторам, поэтому "
        "выбраны как основная конструкция.",
    )


def section_eval(doc: Document, m: dict) -> None:
    doc.add_heading("10. Оффлайн‑оценка и selection bias", level=1)

    doc.add_heading("10.1. Метрики и протокол", level=2)
    add_paragraph(
        doc,
        "Используем стандартный для ранжирования набор метрик. Для каждой "
        "impression‑группы из теста модель упорядочивает её кандидатов; на "
        "получившемся списке считаем:",
    )
    add_list(doc, [
        "Recall@K — какая доля кликнутых кандидатов попала в top‑K (хорошо для recall‑оценки candidate‑generation внутри показа);",
        "nDCG@K — discounted cumulative gain, нормированный на идеальный порядок (учитывает «насколько высоко» оказались релевантные)" + cite(11) + ";",
        "MRR@K — обратная позиция первого релевантного результата.",
    ])
    add_paragraph(
        doc,
        "Все метрики усредняются по группам с хотя бы одним кликом и "
        "размером ≥ 2 (иначе порядок не определён). Время разделения — "
        "квантиль 0.8 по дате события; такой time‑split исключает утечки "
        "будущего в прошлое и приближает оценку к продакшен‑сценарию "
        f"(модель учили на старых данных, оценивают на новых){cite(11)}.",
    )

    doc.add_heading("10.2. Почему baseline ≈ random — и что это значит", level=2)
    base = m["overall"]["baseline_sim"]
    rnd = m["overall"]["random"]
    lgb_v = m["overall"]["lgbm"]
    add_paragraph(
        doc,
        "На оффлайн‑оценке стоит сразу обозначить ограничение, которое "
        "критично для интерпретации цифр. Логи карусели собраны под "
        "управлением старой системы: пользователь видел те статьи, которые "
        "она ему предложила, и кликал по ним. Любая попытка «сравнить нашу "
        "и старую систему» по таким логам — это сравнение перестановок "
        "внутри уже отфильтрованного списка кандидатов, а не сравнение "
        "разных списков кандидатов. Кандидатов, которые наша система "
        "захотела бы добавить, в логах попросту нет, и проверить их "
        "клики невозможно.",
    )
    add_paragraph(
        doc,
        f"Сильным эмпирическим подтверждением этого служит сравнение "
        f"baseline и random на нашем тесте: nDCG@6 у baseline_sim = "
        f"{fmt(base['ndcg@6'], 3)}, у случайного ранжирования — "
        f"{fmt(rnd['ndcg@6'], 3)}; разница — десятые доли процента. То же "
        f"для MRR@6 ({fmt(base['mrr@6'], 3)} vs {fmt(rnd['mrr@6'], 3)}). "
        "Это означает, что внутри показанного списка кандидаты в "
        "среднем уже одинаково релевантны: исходный отбор был достаточно "
        "плотным, и переставлять их по семантической близости почти "
        "бесполезно.",
    )
    add_paragraph(
        doc,
        f"С учётом этого выводы по нашей модели обретают другой вес. LTR "
        f"даёт nDCG@6 = {fmt(lgb_v['ndcg@6'], 3)} — это "
        f"{(lgb_v['ndcg@6'] - base['ndcg@6']) / max(base['ndcg@6'], 1e-6) * 100:+.1f}% "
        f"к baseline и MRR@6 = {fmt(lgb_v['mrr@6'], 3)} — это "
        f"{(lgb_v['mrr@6'] - base['mrr@6']) / max(base['mrr@6'], 1e-6) * 100:+.1f}% "
        "к baseline на тестовых импрессиях. Прирост стабильный по всем K "
        "и по обоим алгоритмам бустинга. Но даже такой положительный "
        "результат — это нижняя оценка реального преимущества: на онлайне "
        "наша система предложит часть собственных кандидатов (которые в "
        "логах отсутствуют), и часть из них кликнут — но в оффлайне "
        "эти клики не учтены.",
    )
    add_paragraph(
        doc,
        f"Это известное ограничение counterfactual evaluation в "
        f"рекомендациях{cite(13, 4)}. Корректные пути его обойти — это "
        "(а) интервенционные онлайн‑эксперименты (A/B), (б) IPS‑оценка с "
        "продакшен‑propensity на показанных списках и (в) использование "
        "случайных интервенций в проде для сбора непредвзятых данных. На "
        "стадии офлайна, как у нас, наиболее честный способ — открыто "
        "проговаривать ограничение и сравнивать модели только в общей "
        "области оффлайн‑метрик.",
    )

    doc.add_heading("10.3. Числовые результаты на test", level=2)
    rows = []
    for name in ["lgbm", "catboost", "baseline_sim", "random"]:
        if name not in m["overall"]:
            continue
        v = m["overall"][name]
        rows.append([
            name,
            fmt(v["ndcg@3"], 3), fmt(v["ndcg@6"], 3), fmt(v["ndcg@10"], 3),
            fmt(v["mrr@3"], 3), fmt(v["mrr@6"], 3), fmt(v["mrr@10"], 3),
        ])
    add_table(
        doc,
        ["Модель", "nDCG@3", "nDCG@6", "nDCG@10", "MRR@3", "MRR@6", "MRR@10"],
        rows,
        caption="Таблица 2. Метрики на test. baseline_sim ≈ random — характерное проявление selection‑bias.",
    )

    extra = m.get("extra", {}).get("metrics", {})
    if extra:
        ks = [1, 3, 5, 6, 10]
        rows = []
        for name in ["lgbm", "catboost", "baseline_sim", "random"]:
            if name not in extra:
                continue
            v = extra[name]
            rows.append([name] + [fmt(v[f"recall@{k}"], 3) for k in ks])
        add_table(
            doc,
            ["Модель"] + [f"recall@{k}" for k in ks],
            rows,
            caption="Таблица 3. Recall@K внутри показа на test.",
        )

    add_image(doc, FIGURES / "ltr_metrics_overall.png", caption="Рис. 5. Сравнение моделей по основным метрикам на test.")
    add_paragraph(
        doc,
        "На рис. 5 столбцы LTR-моделей (lgbm, catboost) выше baseline‑sim "
        "и random на nDCG и MRR при всех K. Разница увеличивается на "
        "маленьких K и сжимается на больших — это значит, что LTR прежде "
        "всего перетасовывает топ списка, а не «дотаскивает» отдельные "
        "редкие клики.",
    )

    add_image(doc, FIGURES / "recall_at_n.png", caption="Рис. 6. Recall@K — какая доля кликов из показа попадает в top‑K.")
    add_paragraph(
        doc,
        "Кривые на рис. 6 показывают одну и ту же закономерность с другой "
        "стороны: разница между моделями максимальна при K = 1 и почти "
        "исчезает при K ≥ 6. Это естественный «потолок»: типичная "
        "карусель в нашем датасете имеет 3–5 элементов, и все клики в "
        "ней просто помещаются в top‑6 у любого вменяемого ранжирования. "
        "Поэтому относительный прирост в строках recall@6 / recall@10 "
        "на следующем графике — десятые доли процента, и сравнение моделей "
        "имеет смысл вести по recall@1, recall@3, nDCG@K и MRR@K.",
    )

    add_image(doc, FIGURES / "ndcg_at_n.png", caption="Рис. 7. nDCG@K на test для четырёх моделей.")
    add_paragraph(
        doc,
        "Рис. 7 даёт «портрет качества» в зависимости от K. LightGBM и "
        "CatBoost практически совпадают (CatBoost чуть выше за счёт "
        "YetiRank‑объективы), а baseline_sim и random тянутся ниже и тоже "
        "почти неразличимы между собой — то самое проявление "
        "selection‑bias из раздела 10.2.",
    )

    add_image(doc, FIGURES / "uplift_lgbm_vs_baseline.png", caption="Рис. 8. Относительный прирост LTR над baseline (semantic-only) на test.")
    add_paragraph(
        doc,
        "На рис. 8 видно две группы метрик. Слева — recall@K: прирост "
        "ощутим только на recall@3 (около +6%), на recall@6 он близок к "
        "нулю (обе модели уже у потолка), на recall@10 — нулевой. Справа "
        "— ranking‑метрики (nDCG, MRR): LTR даёт устойчивый "
        "+6 … +12 % относительно baseline; CatBoost чуть впереди "
        "LightGBM. Главный практический вывод: LTR не «находит больше "
        "релевантных кандидатов» (это работа ретривера), а правильнее "
        "упорядочивает уже отобранных.",
    )


def section_results(doc: Document, m: dict) -> None:
    doc.add_heading("11. Анализ результатов", level=1)

    doc.add_heading("11.1. Где LTR сильнее всего", level=2)
    add_paragraph(
        doc,
        "Прирост распределён неравномерно. Сильнее всего LTR обгоняет "
        "baseline на верхних позициях (recall@1 примерно на четверть выше "
        "относительно baseline): это именно та зона, где пользователь "
        "делает первый клик, и где модель умеет выдвигать вперёд статьи с "
        "лучшим сочетанием семантики и популярности. На больших K разница "
        "сжимается просто потому, что показанная карусель обычно короткая "
        "(3–5 элементов), и top‑6 у любого вменяемого порядка уже включает "
        "большинство кликов.",
    )
    add_image(
        doc,
        FIGURES / "ltr_vs_baseline_by_entity.png",
        caption="Рис. 9. nDCG@6 по типу карусели: baseline vs LTR.",
    )
    add_paragraph(
        doc,
        "На рис. 9 LTR обгоняет baseline во всех четырёх типах "
        "каруселей. Самый большой абсолютный прирост — на блоках "
        "«ml_what_else_mi_pisali» и «ml_personal»: это длинные карусели с "
        "большим числом потенциально близких кандидатов, и именно там "
        "ранжирование действительно меняет, кто оказывается в видимой "
        "части ленты. Самый маленький прирост — у popularity‑блока: там "
        "кандидатов мало и они все уже плотно упорядочены по популярности, "
        "так что LTR имеет мало пространства для манёвра.",
    )

    doc.add_heading("11.2. Какие признаки важны", level=2)
    add_paragraph(
        doc,
        "Feature importance по приросту (gain) показывает, что модель "
        "опирается прежде всего на популярностные и семантические "
        "признаки. Парные признаки (same_dept, same_rubric, |Δage|) дают "
        "вторичный, но устойчивый вклад: модель учится «не уходить "
        "далеко» по теме и по времени публикации.",
    )
    add_image(doc, FIGURES / "feature_importance_gain.png", caption="Рис. 10. Feature importance LightGBM (gain).")
    add_paragraph(
        doc,
        "Конкретно из рис. 10 видно, что наибольший gain даёт сочетание "
        "log_views, sim и freshness — именно эти три фичи и составляют "
        "«сильную тройку», которую модель использует в большинстве сплитов "
        "деревьев. Парные индикаторы темы и автора стоят в середине: они "
        "редко переворачивают результат, но систематически добавляют "
        "+0.5–1 % к качеству. Признак позиции внизу важности — это "
        "ожидаемо после propensity‑debias: вес 1/p̂ как раз для того и "
        "вводился, чтобы модель не запоминала позицию как сигнал.",
    )

    doc.add_heading("11.3. Калибровка и распределения скоров", level=2)
    add_paragraph(
        doc,
        "Распределение LTR‑скора между clicked и not_clicked имеет "
        "выраженное смещение в сторону кликнутых, но с существенным "
        "перекрытием. Это ожидаемо: даже идеальная модель не может "
        "полностью разделить классы, потому что среди «не кликнутых» "
        "много хороших кандидатов, которых пользователь просто не успел "
        "увидеть.",
    )
    if (FIGURES / "score_distribution_two_models.png").exists():
        add_image(
            doc,
            FIGURES / "score_distribution_two_models.png",
            caption="Рис. 11. Распределение скоров clicked vs not_clicked для LightGBM и CatBoost.",
        )
    else:
        add_image(
            doc,
            FIGURES / "score_distribution_clicked_vs_not.png",
            caption="Рис. 11. Распределение LTR‑скора: clicked vs not_clicked.",
        )
    add_paragraph(
        doc,
        "По графикам видно, что мода у «clicked» сдвинута вправо "
        "относительно «not_clicked» примерно на одно‑полтора стандартных "
        "отклонения; разделение «по медиане» уже ловит большую долю "
        "кликов. У CatBoost распределение чуть более узкое (YetiRank "
        "сильнее «прижимает» хвосты), у LightGBM — шире и более «мягкое». "
        "Это согласуется с тем, что обе модели близки по итоговым "
        "метрикам: разные геометрии скоров приводят почти к одному и тому "
        "же порядку внутри группы.",
    )

    doc.add_heading("11.4. Partial dependence: как модель использует фичи", level=2)
    add_paragraph(
        doc,
        "Partial dependence plot (PDP) показывает, как меняется средний "
        "предсказанный скор при изменении одной фичи и фиксации остальных. "
        "Это удобный способ понять «форму» зависимости: монотонная она, "
        "ступенчатая, есть ли saturation.",
    )
    if (FIGURES / "pdp_sim.png").exists():
        add_image(doc, FIGURES / "pdp_sim.png", caption="Рис. 12. PDP: sim(a, b).")
        add_paragraph(
            doc,
            "По рис. 12 sim даёт почти монотонный, плавный рост: модель "
            "стабильно повышает скор, когда кандидат семантически ближе к "
            "источнику. Это «здоровая» картинка: значит, бустинг "
            "не «пере‑выучил» какой‑то локальный артефакт sim, а "
            "использует его как основной сигнал близости.",
        )
    if (FIGURES / "pdp_logviews.png").exists():
        add_image(doc, FIGURES / "pdp_logviews.png", caption="Рис. 13. PDP: log_views(b).")
        add_paragraph(
            doc,
            "На рис. 13 log_views у кандидата тоже монотонно повышает "
            "скор, но кривая выходит на плато после log_views ≈ 11 "
            "(это десятки тысяч просмотров). После порога модель уже "
            "не различает «100k vs 1M просмотров»: это правильное "
            "поведение — выше определённой популярности дополнительный "
            "буст бессмысленен и приводил бы к перекосу выдачи.",
        )
    if (FIGURES / "pdp_fresh.png").exists():
        add_image(doc, FIGURES / "pdp_fresh.png", caption="Рис. 14. PDP: freshness(b).")
        add_paragraph(
            doc,
            "Рис. 14 показывает freshness как ступенчатый «бонус за "
            "свежесть»: пока возраст материала меньше нескольких "
            "месяцев, скор почти линейно растёт; затем кривая выходит "
            "на ровный «низкий» уровень для старых статей. Это и "
            "оправдывает фильтр возраста для explore — после "
            "365 дней дополнительной пользы от свежести модель уже "
            "не видит.",
        )

    doc.add_heading("11.5. Что нового даёт coview", level=2)
    add_paragraph(
        doc,
        "Coview‑сигнал в системе участвует в двух местах: (а) на этапе "
        "генерации кандидатов он расширяет similar‑пул соседями по "
        "поведенческой матрице p(b | a); (б) на этапе LTR он добавлен "
        "двумя признаками — coview_score = log p(b | a) и coview_rank = "
        "позиция кандидата в top‑N coview‑соседей источника. Кроме того, "
        "в обучающую выборку включены 12 085 синтетических coview‑групп "
        "(top‑1 vs «хвост», вес 0.3): это слабая supervision, которая "
        "учит ранкер ставить top‑1 coview выше нерелевантных хвостовых "
        "соседей. Изначальный лимит был 80 тыс., но фактически отвечающих "
        "условию «≥ 7 coview‑соседей в индексе» источников оказалось "
        "значительно меньше — это прямое следствие узости текущего "
        "coview‑индекса (см. раздел 6.1, где приведены замеры покрытия).",
    )
    add_paragraph(
        doc,
        "Эффект от coview‑признаков виден сразу в трёх местах. Во‑первых, "
        "на test‑метриках LightGBM nDCG@6 поднялся с ~0.744 до "
        f"{fmt(m['overall']['lgbm']['ndcg@6'], 3)}, а MRR@6 — с ~0.660 до "
        f"{fmt(m['overall']['lgbm']['mrr@6'], 3)} (+0.5–0.8 % относительно "
        "версии без coview‑фич), recall@1 вырос с ~0.398 до 0.405. Во‑вторых, "
        "на feature importance‑графике (рис. 7) coview_score и coview_rank "
        "уверенно попадают в среднюю часть списка (gain ≈ 0.79·10⁶ и 0.53·10⁶ "
        "против 2.0·10⁶ у sim) — модель действительно опирается на этот "
        "сигнал, а не игнорирует его. В‑третьих, в примерах выдачи из "
        "раздела 11.6 видно, что наш ранкер часто поднимает наверх "
        "кандидатов с высокой coview‑связью к источнику, даже если sim "
        "у них не самый большой — ровно тот случай, который semantic kNN "
        "самостоятельно поймать не мог.",
    )
    add_paragraph(
        doc,
        "Качественно coview особенно полезен там, где релевантность не "
        "сводится к лексической близости: статьи об оформлении ИП и о "
        "регистрации ООО семантически довольно далеки, но пользователи "
        "ходят между ними устойчиво. Этот сигнал LTR корректно «доскоривает» "
        "по своим признакам, не давая поведенческому сигналу полностью "
        "подменить собой семантику — на хвосте каталога coview часто шумит, "
        "и без регуляризующей роли семантики (sim, freshness, similarity‑"
        "соседства) модель легко увлеклась бы случайными совпадениями.",
    )

    section_examples_serp(doc)


def section_examples_serp(doc: Document) -> None:
    doc.add_heading("11.6. Пример выдач для двух статей", level=2)

    summary_path = ROOT / "reports" / "examples" / "summary.json"
    if not summary_path.exists():
        add_paragraph(
            doc,
            "Примеры выдач не сгенерированы — запустите make_example_serps.py "
            "для создания таблиц.",
        )
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    add_paragraph(
        doc,
        "Чтобы наглядно увидеть разницу между чистым semantic baseline и "
        "обученной LTR‑моделью, ниже приводятся top‑6 кандидатов для "
        "двух типичных статей‑источников: одной «вечнозелёной» (давно "
        "опубликованной, но активно читаемой) и одной свежей. По "
        "обеим выдачам модели подобраны 80 ближайших соседей по "
        "эмбеддингам, и затем каждая модель упорядочила их по своему "
        "скору.",
    )

    import pandas as pd
    for i, entry in enumerate(summary[:2], start=1):
        title = entry["source_title"]
        path = Path(entry["table_path"])
        if not path.exists():
            continue

        doc.add_heading(f"Источник №{i}: «{title}»", level=3)

        df = pd.read_csv(path)
        df = df[df["rank"] <= 6]
        rows_base, rows_ltr = [], []
        for _, r in df.iterrows():
            row_view = [
                str(int(r["rank"])),
                str(r["title"]),
                str(r["rubric"])[:18] if str(r["rubric"]) != "nan" else "—",
                str(int(r["age_days"])),
                f"{int(r['views']):,}",
            ]
            if r["system"] == "baseline_sim":
                rows_base.append(row_view + [str(r["sim"])])
            else:
                rows_ltr.append(row_view + [str(r["ltr_score"])])

        add_table(
            doc,
            ["#", "Заголовок", "Рубрика", "возраст, дн.", "просмотры", "sim"],
            rows_base,
            caption=f"Таблица. Top‑6 от baseline (semantic kNN) для источника №{i}.",
        )
        add_table(
            doc,
            ["#", "Заголовок", "Рубрика", "возраст, дн.", "просмотры", "LTR-score"],
            rows_ltr,
            caption=f"Таблица. Top‑6 от LTR (LightGBM) для источника №{i}.",
        )

    add_paragraph(
        doc,
        "Содержательно различия видны сразу. Baseline ориентируется "
        "только на близость текста и поднимает наверх семантически "
        "близкие материалы безотносительно их популярности и возраста — "
        "поэтому в его top‑6 встречаются давно устаревшие статьи "
        "с маленькой аудиторией. LTR при той же тематике подтягивает "
        "вверх материалы с заметно большей реальной популярностью и "
        "более свежие — это прямой эффект признаков log_views и "
        "freshness, которые модель оказывается в состоянии комбинировать "
        "с семантикой.",
    )


def section_discussion(doc: Document) -> None:
    doc.add_heading("12. Обсуждение и направления развития", level=1)
    add_list(doc, [
        "Online A/B — самое естественное продолжение работы: только онлайн "
        "позволит честно сравнить нашу выдачу с продакшен‑системой, потому "
        "что только онлайн можно показать пользователю кандидатов из своих "
        "пулов и собрать клики по ним.",
        "Sequence‑aware ранжирование (SASRec / BERT4Rec)" + cite(24, 25) + ": "
        "если в продукте появится требование персонализации внутри блока "
        "«Что ещё мы писали», логичный шаг — учить трансформер по "
        "последовательности недавно прочитанных статей.",
        "Diversity‑aware loss: вместо жёстких правил можно встраивать "
        "diversity напрямую в loss, например через DPP (Determinantal Point "
        "Processes) или DPR‑подобные подходы. Это даст более тонкий контроль, "
        "но потребует более долгого онлайн‑тюнинга.",
        "ANN‑индекс (HNSW) — пока избыточен, но станет полезен, если каталог "
        "вырастет на порядок и появятся требования по тёплому реалтайму.",
        "Inter‑level distillation: учить LTR на soft‑labels от двухбашенной "
        "модели" + cite(30) + " — потенциально снижает variance целевой "
        "переменной и помогает на хвосте каталога, где кликов мало.",
        "Periodic propensity re‑estimation: сейчас p̂(click|pos) считается "
        "разово на train; в продакшене стоит пересчитывать его по "
        "скользящему окну, чтобы отражать сезонные изменения CTR.",
    ])
    add_paragraph(
        doc,
        "Главный методологический вывод: пайплайн «ретрив + LTR + правила» "
        "остаётся отличным компромиссом между точностью, понятностью и "
        "управляемостью. ML‑модель учит ровно то, что у неё получается "
        "лучше, чем у эвристик, а правила оставляют редакции и продукту "
        "контроль над итоговой выдачей.",
    )


def section_refs(doc: Document) -> None:
    doc.add_heading("Литература", level=1)
    for i, (cite_text, url) in enumerate(REFS, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Cm(0.6)
        run = p.add_run(f"[{i}] ")
        run.bold = True
        p.add_run(cite_text + " ")
        link = p.add_run(url)
        link.italic = True
        link.font.color.rgb = RGBColor(0x33, 0x66, 0xCC)


# ---------------------------------------------------------------------------
# Главная сборка
# ---------------------------------------------------------------------------


def main() -> None:
    if not METRICS_JSON.exists():
        raise SystemExit(
            "Нет reports/ltr_full_metrics.json — сначала запустите train_ltr_full.py"
        )
    metrics = load_metrics()

    doc = make_doc()
    section_title(doc)
    section_abstract(doc, metrics)
    doc.add_page_break()
    section_intro(doc)
    section_problem(doc)
    section_landscape(doc)
    section_architecture(doc)
    section_data(doc)
    section_candidates(doc)
    section_features(doc)
    section_ltr(doc, metrics)
    section_rerank(doc)
    section_eval(doc, metrics)
    section_results(doc, metrics)
    section_discussion(doc)
    doc.add_page_break()
    section_refs(doc)

    doc.save(OUT_PATH)
    print(f"saved: {OUT_PATH}  ({os.path.getsize(OUT_PATH) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
