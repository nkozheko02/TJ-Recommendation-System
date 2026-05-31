"""Конвертер `.py → .ipynb` для удобного просмотра в Google Colab.

Разбивает Python‑файл по top‑level конструкциям (импорты, константы, функции,
классы) — каждая получает свою cell. Перед каждой функцией/классом вставляется
markdown‑заголовок с именем, чтобы в Colab было удобно листать через outline.

Использование:
    python scripts/py_to_ipynb.py serve_carousel.py
    python scripts/py_to_ipynb.py serve_carousel.py train_ltr_full.py ...
    python scripts/py_to_ipynb.py --all   # все 5 ключевых файлов

Выход: notebooks/<имя>.ipynb
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks"

DEFAULT_FILES = [
    "serve_carousel.py",
    "train_ltr_full.py",
    "eval_ltr_i2i.py",
    "extra_analytics.py",
    "train_eval_no_coview_i2i.py",
]

# Файлы, для которых нужен исходный 2.2GB parquet-лог
# (помечаем чтобы bootstrap-ячейка предупредила).
NEEDS_LOGS = {
    "train_ltr_full.py",
    "eval_ltr_i2i.py",
    "train_eval_no_coview_i2i.py",
    "extra_analytics.py",
}

# Демо-ячейка добавляется в конец ноутбука (после if __name__ == "__main__"),
# чтобы можно было запустить inference прямо в Colab без CLI.
DEMO_APPENDIX = {
    "serve_carousel.py": (
        "## Демо: одна карусель прямо в ноутбуке\n",
        '''# Готовим контекст один раз (загрузка статей + эмбеддингов + модели + coview ~30 сек).
ctx = build_context()

# Берём первый попавшийся article_id из каталога как пример.
example_id = ctx.art.df["article_id"].iloc[0]
print("source:", example_id, "—", ctx.art.df["article_base__title"].iloc[0])

# Сборка карусели i2i. Возвращает DataFrame из 12 рекомендаций.
carousel = serve_one(ctx, example_id)
carousel[[
    "position", "article_id", "title", "score",
    "scoring_mode", "pool", "rubric", "author",
]]
''',
    ),
}


COLAB_BOOTSTRAP_TEMPLATE = '''# === Colab Bootstrap (запускается один раз в начале сессии) ===
#
# Что делает:
#   1. Определяет, что мы в Google Colab.
#   2. Монтирует Google Drive (нужно подтвердить в всплывающем окне).
#   3. Ищет архив `tj_recs_colab_inference.zip` в Drive и распаковывает в /content/tj-recs.
#   4. Делает chdir в распакованную папку и добавляет её в sys.path.
#   5. Доставляет недостающие пакеты (lightgbm и т.п.).
#
# Локально (вне Colab) ячейка просто переключает cwd в корень репо и не делает install.

import os, sys

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    BUNDLE_NAME = "tj_recs_colab_inference.zip"
    BUNDLE_PATH = f"/content/drive/MyDrive/{BUNDLE_NAME}"
    WORKDIR = "/content/tj-recs"

    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"Не нашёл {BUNDLE_PATH}. Положи tj_recs_colab_inference.zip в корень "
            f"My Drive (или поменяй BUNDLE_PATH в этой ячейке)."
        )

    if not os.path.isdir(WORKDIR) or not os.listdir(WORKDIR):
        os.makedirs(WORKDIR, exist_ok=True)
        import zipfile
        print(f"Распаковываю {BUNDLE_PATH} → {WORKDIR} (~1-2 мин)…")
        with zipfile.ZipFile(BUNDLE_PATH) as z:
            z.extractall(WORKDIR)
        print("OK")
    else:
        print(f"{WORKDIR} уже распакован — пропускаю.")

    os.chdir(WORKDIR)
    if WORKDIR not in sys.path:
        sys.path.insert(0, WORKDIR)

    print("Ставлю зависимости…")
    %pip install -q lightgbm catboost pyarrow
__LOGS_NOTE__
else:
    # Локальный запуск — переключаемся в корень репо.
    LOCAL_REPO = "/Users/nkozheko/T-Ж рекомендации"
    if os.path.isdir(LOCAL_REPO):
        os.chdir(LOCAL_REPO)
        if LOCAL_REPO not in sys.path:
            sys.path.insert(0, LOCAL_REPO)
        print("Локальный режим, cwd:", os.getcwd())
    else:
        print("Локальный путь не найден — продолжаем в текущей cwd:", os.getcwd())

print("Готово. sys.path[0] =", sys.path[0])
'''

LOGS_NOTE = '''
    # Для этого ноутбука нужен исходный лог:
    LOGS_PATH = "/content/drive/MyDrive/tj_session_w_target_full.parquet"
    if not os.path.exists(LOGS_PATH):
        print(
            "⚠ ВНИМАНИЕ: этот ноутбук обучает/оценивает модель и требует\\n"
            f"   {LOGS_PATH} (~2.2 GB). Залей его в My Drive отдельно.\\n"
            "   Иначе сработают только ячейки, которые не вызывают main()."
        )
    else:
        # Переопределяем путь — в train_ltr_full.py LOGS_PARQUET захардкожен на mac.
        os.environ["TJ_LOGS_PARQUET"] = LOGS_PATH'''


def make_bootstrap_cells(filename: str) -> list[dict]:
    """Возвращает 2 cells: markdown-заголовок и code-ячейка с bootstrap."""
    needs_logs = filename in NEEDS_LOGS
    logs_note = LOGS_NOTE if needs_logs else ""
    bootstrap_src = COLAB_BOOTSTRAP_TEMPLATE.replace("__LOGS_NOTE__", logs_note)
    md = (
        "## Colab Bootstrap\n\n"
        "Запусти эту ячейку **первой** при работе в Colab. Она смонтирует Google "
        "Drive, распакует `tj_recs_colab_inference.zip` и поставит зависимости. "
        "Локально просто переключит cwd в репозиторий.\n"
    )
    return [_md_cell(md), _code_cell(bootstrap_src)]


def _md_cell(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def _code_cell(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


def _slice_source(src_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "".join(src_lines[start:end])


def split_into_cells(src: str, filename: str) -> list[dict]:
    """Парсит файл и возвращает список cells.

    Логика:
      - Заголовок и Colab Bootstrap (md + code) — в самом начале.
      - Module docstring → md-cell.
      - Все непрерывные импорты и `from ...` → одна code-cell.
      - Группа top-level присваиваний (CONSTANTS) → одна code-cell.
      - Каждая FunctionDef / AsyncFunctionDef / ClassDef → markdown-заголовок
        с именем и docstring, потом code-cell с телом.
      - Главный блок `if __name__ == "__main__"` → отдельная code-cell в конце.
      - Опционально — demo-ячейка из DEMO_APPENDIX.
    """

    src_lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    cells: list[dict] = []

    # Шапка с описанием файла
    intro = f"# `{filename}`\n\n"
    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        intro += mod_doc + "\n"
    cells.append(_md_cell(intro))

    # Bootstrap для Colab — сразу после шапки
    cells.extend(make_bootstrap_cells(filename))

    # Skip docstring при последующей нарезке
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    # Группируем подряд идущие импорты и присваивания в один cell
    buffer: list[ast.AST] = []
    buffer_kind: str | None = None  # "imports" | "assigns"

    def flush_buffer():
        nonlocal buffer, buffer_kind
        if not buffer:
            return
        text = "".join(_slice_source(src_lines, n) for n in buffer)
        # обрезаем хвостовые пустые строки
        text = text.rstrip() + "\n"
        if buffer_kind == "imports":
            cells.append(_md_cell("## Импорты\n"))
        elif buffer_kind == "assigns":
            cells.append(_md_cell("## Константы и настройки\n"))
        cells.append(_code_cell(text))
        buffer = []
        buffer_kind = None

    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if buffer_kind != "imports":
                flush_buffer()
                buffer_kind = "imports"
            buffer.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if buffer_kind != "assigns":
                flush_buffer()
                buffer_kind = "assigns"
            buffer.append(node)
        elif isinstance(node, ast.If) and _is_main_guard(node):
            flush_buffer()
            text = _slice_source(src_lines, node).rstrip() + "\n"
            cells.append(_md_cell("## Точка входа\n"))
            cells.append(_code_cell(text))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_buffer()
            _emit_function(cells, src_lines, node, kind="def")
        elif isinstance(node, ast.ClassDef):
            flush_buffer()
            _emit_function(cells, src_lines, node, kind="class")
        else:
            # любые другие выражения (try/with на верхнем уровне) — как есть
            if buffer_kind == "assigns":
                buffer.append(node)
            else:
                flush_buffer()
                text = _slice_source(src_lines, node).rstrip() + "\n"
                cells.append(_code_cell(text))

    flush_buffer()

    # Demo-аппендикс (например, готовый serve_one() запуск)
    if filename in DEMO_APPENDIX:
        md_text, code_text = DEMO_APPENDIX[filename]
        cells.append(_md_cell(md_text))
        cells.append(_code_cell(code_text))

    return cells


def _is_main_guard(node: ast.If) -> bool:
    """Проверяет, что if это `if __name__ == '__main__':`."""
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _emit_function(cells: list[dict], src_lines: list[str], node: ast.AST, *, kind: str) -> None:
    name = node.name
    doc = ast.get_docstring(node) if hasattr(node, "body") else None
    label = "Функция" if kind == "def" else "Класс"
    md = f"## {label} `{name}`\n"
    if doc:
        # уберём отступы и оставим лишь первый абзац для читаемости
        first_para = doc.strip().split("\n\n", 1)[0].strip()
        md += f"\n> {first_para}\n"
    cells.append(_md_cell(md))

    text = _slice_source(src_lines, node).rstrip() + "\n"
    cells.append(_code_cell(text))


def build_notebook(cells: list[dict]):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    nb.cells = cells
    return nb


def convert(py_path: Path) -> Path:
    src = py_path.read_text(encoding="utf-8")
    cells = split_into_cells(src, py_path.name)
    nb = build_notebook(cells)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (py_path.stem + ".ipynb")
    nbf.write(nb, out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="Python‑файлы для конвертации (относительно корня репо)")
    ap.add_argument("--all", action="store_true", help="конвертировать все 5 ключевых файлов")
    args = ap.parse_args()

    if args.all or not args.files:
        files = DEFAULT_FILES
    else:
        files = args.files

    for f in files:
        p = ROOT / f
        if not p.exists():
            print(f"[skip] {f}: file not found")
            continue
        out = convert(p)
        nb = nbf.read(out, as_version=4)
        print(f"  {f}  ->  {out.relative_to(ROOT)}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
