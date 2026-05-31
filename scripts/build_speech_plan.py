"""Собирает короткий план устного рассказа для предзащиты в docx.

Два сегмента:
  1. Архитектура → Кандидаты → LTR → Фичи → Метрики → Карусель  (~4 мин)
  2. Стратегия → Будущее (от слайда «Стратегия» до «Варианта будущего»)  (~2 мин)

Файл: reports/speech_plan_predefence.docx
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "speech_plan_predefence.docx"

# --- содержимое плана -------------------------------------------------------

SLIDES = [
    {
        "title": "Слайд 24 — Архитектура (общая схема)",
        "speech": (
            "Архитектура решения — двухпуловая. Из 100 тысяч статей каталога "
            "мы под каждый источник за миллисекунды отбираем ~200 кандидатов, "
            "ранжируем их LTR-моделью и собираем итоговую карусель из 12 позиций "
            "с продуктовыми правилами. Каждый из этих трёх блоков — Retrieval, "
            "Ranking, Reranking — решает свою чёткую задачу, и сейчас я кратко "
            "расскажу про каждый."
        ),
        "transition": "Начнём с того, откуда берутся кандидаты. → слайд 26",
        "duration": "~30 сек",
    },
    {
        "title": "Слайд 26 — Генерация кандидатов (два пула)",
        "speech": (
            "У нас два независимых пула. Similar — это похожие статьи: top-100 "
            "по cosine-близости текстовых эмбеддингов плюс top-50 поведенческих "
            "соседей из coview-индекса. Cosine ловит «то же самое перефразированное», "
            "coview — «то, что обычно читают после», даже если оно семантически "
            "далеко. Explore — пул разнообразия: топ по quality + trend внутри "
            "департамента и глобально, чтобы карусель не зацикливалась на одной "
            "теме. Они дают суммарно около 200 кандидатов на источник."
        ),
        "transition": "Теперь нужно из этих 200 выбрать 12. Это делает LTR. → слайд 28",
        "duration": "~45 сек",
    },
    {
        "title": "Слайды 28–29 — Ранжирование (LTR)",
        "speech": (
            "Ранжирует двухголовый LTR — LightGBM LambdaRank и CatBoost YetiRank. "
            "На вход — 17 признаков: семантика, поведение, популярность, свежесть, "
            "кросс-характеристики. Обучаем на 22 миллионах строк логов "
            "показов/кликов с поправкой на позиционный bias через propensity "
            "weighting. Если у источника одновременно нет coview-соседей и статья "
            "совсем свежая — переключаемся на эвристический скор "
            "«sim + popularity + freshness + rubric». Это узкая страховка под "
            "cold-start — на полном каталоге она затрагивает 27 статей из 104 тысяч."
        ),
        "transition": "Что модель в итоге считает важным — на следующем слайде. → слайд 30",
        "duration": "~50 сек",
    },
    {
        "title": "Слайд 30 — Важность признаков",
        "speech": (
            "Топ-5 — sim, src_log_views, src_like_rate, разница возрастов и "
            "свежесть источника. Coview-признаки — на 13–14 местах из 17, но они "
            "дают свой вклад: без них recall@1 падает с 0.41 до 0.40. Главный "
            "вывод — модель опирается на смесь семантики, статистики и поведения, "
            "а не на один источник."
        ),
        "transition": "Теперь — сами метрики. → слайд 31",
        "duration": "~20 сек",
    },
    {
        "title": "Слайд 31 — Результаты LTR",
        "speech": (
            "Главный замер — на i2i-тесте, 20 тысяч валидных групп. LightGBM "
            "поднимает Recall@1 с 31 до 41 % — то есть в 41 % показов кликнутая "
            "статья теперь стоит первой, а не где-то ниже. По nDCG@6 — рост с "
            "0.70 до 0.75, по MRR@6 — с 0.60 до 0.67. CatBoost чуть впереди. "
            "Это нижняя оценка реального онлайн-эффекта — selection bias логов "
            "снизу подрезает разницу."
        ),
        "transition": "Финальный шаг — из отранжированных 200 кандидатов собрать 12 видимых. → слайд 32",
        "duration": "~40 сек",
    },
    {
        "title": "Слайды 32–33 — Формирование карусели",
        "speech": (
            "Карусель — это не просто top-12 по скору. Это 9 similar + 3 explore, "
            "причём explore идут на фиксированных позициях 4, 7 и 10, чтобы "
            "разнообразие было распределено по всей выдаче, а не свалено в хвост. "
            "Поверх — продуктовые правила: не более 2 статей одного автора, не "
            "более 6 одной рубрики, дедуп по нормализованному заголовку, фильтр "
            "старше года, блок-лист «опасных» кросс-дептовых переходов. Explore "
            "дополнительно семплируется через softmax — чтобы у разных источников "
            "верхние слоты не были одинаковыми. На выходе — стабильные 12 статей "
            "со флагом scoring_mode для мониторинга."
        ),
        "transition": "Так из 100 тысяч статей за один прогон формируется "
                      "персонализированная карусель из 12. Пример выдач — на следующем слайде.",
        "duration": "~45 сек",
    },
]

SLIDES_STRATEGY = [
    {
        "title": "Слайд 40 — Стратегия: Как мы видим Т—Ж будущего",
        "speech": (
            "Мы построили MVP — единую ML-карусель для новичков. Это первый шаг. "
            "Дальше — четыре направления, в которые логично развивать систему, "
            "чтобы Т—Ж стал персональным медиа уже с первой сессии, а не «после "
            "первого захода»."
        ),
        "transition": "Первое направление — научиться узнавать пользователя "
                      "в моменте, без анкет. → слайд 41",
        "duration": "~20 сек",
    },
    {
        "title": "Слайд 41 — Параллельный онбординг",
        "speech": (
            "Сейчас мы у пользователя ничего не спрашиваем — он просто читает. "
            "Идея — собирать сигналы интереса параллельно с чтением: микро-клики, "
            "глубина скролла, время на абзаце, hover на ссылках. Это даёт неявную "
            "«анкету» без отвлекающих pop-up-ов и работает уже с первой страницы."
        ),
        "transition": "Сигналы собираем — теперь нужно их где-то использовать. "
                      "Не только в одной карусели. → слайд 42",
        "duration": "~25 сек",
    },
    {
        "title": "Слайд 42 — Гибкие форматы рекомендаций",
        "speech": (
            "Сегодня рекомендации — одна карусель внизу статьи. Но контекст у "
            "пользователя бывает разный: кому-то лучше боковая колонка, кому-то "
            "— inline-вставка между параграфами, кому-то — bottom-sheet на "
            "мобильном. Идея — ML выбирает не только что показать, но и в каком "
            "формате это будет уместно именно сейчас."
        ),
        "transition": "Каждый формат должен объяснять пользователю «почему именно "
                      "эта статья». → слайд 43",
        "duration": "~25 сек",
    },
    {
        "title": "Слайд 43 — Контекстные подписи",
        "speech": (
            "Гипотеза «краткие описания», которую мы сейчас выводим в A/B, — "
            "это начало. Следующий шаг — подписи, зависящие от контекста: "
            "«продолжение темы», «базовый разбор для новичков», «глубже в "
            "детали для экспертов». Они снижают цену клика: пользователь сразу "
            "видит, зачем ему конкретно эта статья."
        ),
        "transition": "Чтобы такие подписи и форматы были осмысленными — нужна "
                      "модель пользователя. → слайд 44",
        "duration": "~25 сек",
    },
    {
        "title": "Слайд 44 — Идеальное знание о читателе",
        "speech": (
            "В идеале мы хотим прийти к полноценной модели пользователя по "
            "первой же сессии: интент (утилитарный, исследующий, серфинг), "
            "экспертный уровень, тематические предпочтения, эмоциональный тон. "
            "Сейчас всё это либо отсутствует, либо собирается по косвенным "
            "признакам. С этими сигналами наш i2i превращается в полноценный "
            "персональный stream."
        ),
        "transition": "Соберём всё вместе на одном экране. → слайд 45",
        "duration": "~25 сек",
    },
    {
        "title": "Слайд 45 — Вариант будущего",
        "speech": (
            "На этом видео — собранная картинка того, как это могло бы выглядеть "
            "в одном продуктовом сценарии. Лента из нескольких форматов, "
            "контекстные подписи, незаметный онбординг — и в центре адаптивная "
            "карусель, в которую под капотом встроена наша текущая ML-модель. "
            "Каждый из четырёх кирпичей мы можем добавлять инкрементально, не "
            "ломая то, что уже работает."
        ),
        "transition": "На этом основная часть закончена — спасибо. Готов ответить "
                      "на вопросы или показать детали в приложении.",
        "duration": "~20 сек",
    },
]

DO_NOT_SAY = [
    "Точные значения MIN_COVIEW_NEIGHBORS = 3 / MIN_SOURCE_AGE_DAYS = 3 — упомянуть "
    "как «несколько соседей и младше нескольких дней», без чисел.",
    "Формулу score_cov = log p(b|a) — достаточно сказать «вероятность перехода».",
    "CatBoost vs LightGBM по отдельности на каждом слайде — упомянуть один раз и "
    "дальше говорить «LTR-модель».",
    "Цифру «20 781 групп» — округлить до «~20 тысяч».",
    "Детали про propensity_floor = 0.02 — сказать только «с поправкой на position bias».",
]

KEY_MESSAGES = (
    "Двухпуловый retrieval → LTR с 17 фичами и cold-start fallback → продуктовая "
    "сборка с интерливингом."
)


# --- стилизация helpers -----------------------------------------------------


def _set_run(run, *, size: int | None = None, bold: bool = False,
             italic: bool = False, color: tuple[int, int, int] | None = None) -> None:
    if size is not None:
        run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def add_heading_custom(doc: Document, text: str, *, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Helvetica"
        if level == 0:
            _set_run(run, size=22, bold=True, color=(11, 29, 58))
        elif level == 1:
            _set_run(run, size=14, bold=True, color=(31, 111, 235))
        else:
            _set_run(run, size=12, bold=True, color=(40, 40, 40))


def add_para(doc: Document, text: str, *, italic: bool = False,
             color: tuple[int, int, int] | None = None,
             size: int = 11) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Helvetica"
    _set_run(r, size=size, italic=italic, color=color)


def add_kv(doc: Document, key: str, value: str) -> None:
    p = doc.add_paragraph()
    r1 = p.add_run(f"{key}: ")
    _set_run(r1, size=11, bold=True, color=(40, 40, 40))
    r2 = p.add_run(value)
    _set_run(r2, size=11)


# --- сборка документа -------------------------------------------------------


def build() -> None:
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(1.8)
        section.bottom_margin = Cm(1.8)
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)

    add_heading_custom(doc, "План рассказа на предзащиту", level=0)
    sub = doc.add_paragraph()
    r = sub.add_run(
        "Два сегмента  ·  ~6 минут  ·  12 слайдов\n"
        "Часть 1 (~4 мин): Архитектура → Кандидаты → LTR → Фичи → Метрики → Карусель\n"
        "Часть 2 (~2 мин): Стратегия → Параллельный онбординг → Гибкие форматы → "
        "Контекстные подписи → Знание о читателе → Вариант будущего"
    )
    _set_run(r, size=11, italic=True, color=(120, 120, 120))

    add_para(doc, "")

    # =====================================================================
    # ЧАСТЬ 1 — техническая
    # =====================================================================
    add_heading_custom(doc, "Часть 1. Технический рассказ про систему рекомендаций "
                            "(≈ 3.5–4 минуты)", level=1)

    for sl in SLIDES:
        add_heading_custom(doc, sl["title"], level=2)
        add_kv(doc, "Длительность", sl["duration"])

        add_para(doc, "Реплика:", color=(40, 40, 40), size=11)
        add_para(doc, sl["speech"], size=12)

        add_para(doc, f"Переход: {sl['transition']}", italic=True,
                 color=(120, 120, 120), size=10)
        add_para(doc, "")

    # =====================================================================
    # ЧАСТЬ 2 — стратегия
    # =====================================================================
    add_heading_custom(doc, "Часть 2. Видение и стратегия развития "
                            "(≈ 2 минуты)", level=1)
    add_para(
        doc,
        "Этот сегмент — про то, куда логично развивать систему дальше. Цифр "
        "почти нет, тон — визионерский, опираемся на названия слайдов и общие "
        "идеи. Главная мысль для каждого слайда подана одной фразой плюс "
        "одной фразой пояснения.",
        italic=True, color=(120, 120, 120), size=10,
    )
    add_para(doc, "")

    for sl in SLIDES_STRATEGY:
        add_heading_custom(doc, sl["title"], level=2)
        add_kv(doc, "Длительность", sl["duration"])

        add_para(doc, "Реплика:", color=(40, 40, 40), size=11)
        add_para(doc, sl["speech"], size=12)

        add_para(doc, f"Переход: {sl['transition']}", italic=True,
                 color=(120, 120, 120), size=10)
        add_para(doc, "")

    # =====================================================================
    # тайминги
    # =====================================================================
    add_heading_custom(doc, "Тайминги (ориентир)", level=1)

    all_slides = SLIDES + SLIDES_STRATEGY
    table = doc.add_table(rows=len(all_slides) + 4, cols=2)
    table.style = "Light Grid Accent 1"

    hdr = table.rows[0].cells
    hdr[0].text = "Слайд"
    hdr[1].text = "Время"
    for c in hdr:
        for p in c.paragraphs:
            for run in p.runs:
                _set_run(run, size=11, bold=True)

    row_idx = 1

    # Часть 1
    part1_header = table.rows[row_idx].cells
    part1_header[0].text = "ЧАСТЬ 1 — Технический рассказ"
    part1_header[1].text = ""
    for c in part1_header:
        for p in c.paragraphs:
            for run in p.runs:
                _set_run(run, size=10, bold=True, color=(31, 111, 235))
    row_idx += 1

    for sl in SLIDES:
        row = table.rows[row_idx].cells
        row[0].text = sl["title"].split("—")[0].strip()
        row[1].text = sl["duration"]
        for c in row:
            for p in c.paragraphs:
                for run in p.runs:
                    _set_run(run, size=10)
        row_idx += 1

    # Часть 2
    part2_header = table.rows[row_idx].cells
    part2_header[0].text = "ЧАСТЬ 2 — Видение и стратегия"
    part2_header[1].text = ""
    for c in part2_header:
        for p in c.paragraphs:
            for run in p.runs:
                _set_run(run, size=10, bold=True, color=(31, 111, 235))
    row_idx += 1

    for sl in SLIDES_STRATEGY:
        row = table.rows[row_idx].cells
        row[0].text = sl["title"].split("—")[0].strip()
        row[1].text = sl["duration"]
        for c in row:
            for p in c.paragraphs:
                for run in p.runs:
                    _set_run(run, size=10)
        row_idx += 1

    total = table.rows[-1].cells
    total[0].text = "Итого"
    total[1].text = "~5.5–6 мин"
    for c in total:
        for p in c.paragraphs:
            for run in p.runs:
                _set_run(run, size=11, bold=True, color=(11, 29, 58))

    add_para(doc, "")

    # --- ключевые сообщения
    add_heading_custom(doc, "Три ключевых сообщения, которые держать в нитке", level=1)
    add_para(doc, KEY_MESSAGES, size=12)
    add_para(doc, "")

    # --- что не говорить
    add_heading_custom(doc, "Что НЕ говорить вслух", level=1)
    add_para(
        doc,
        "Эти детали есть в материалах и могут быть в кармане для ответа на вопросы, "
        "но в основной нитке утяжелят рассказ.",
        italic=True, color=(120, 120, 120), size=10,
    )
    for item in DO_NOT_SAY:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(item)
        _set_run(r, size=11)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    build()
