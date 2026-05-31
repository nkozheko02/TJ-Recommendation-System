"""Собирает минимальный zip‑бандл для запуска `serve_carousel` в Google Colab.

В бандл попадает только то, что реально нужно для inference карусели:
  - все .py‑файлы из корня (включая `serve_carousel.py`, `train_ltr_full.py`)
  - models/ltr_lgbm.pkl  + ltr_catboost.cbm  + ltr_lgbm_no_coview.pkl
  - coview_cache/coview_index.pkl.gz
  - embeddings_cache/  (memmap + ids.txt + shape.txt)
  - tj_article.csv  (каталог)
  - notebooks/00_demo_serve_carousel.ipynb  (главный демо‑ноутбук)
  - requirements.txt

Что НЕ попадает (слишком тяжело и не нужно для serve):
  - user_articles_embeddings.csv (1.9 GB) — нужен только для тренировки/eval
  - recs_i2i.csv (177 MB) — старый артефакт
  - reports/, catboost_info/, mockups, html, diploma.docx и т.п.

Использование:
    python scripts/prepare_colab_bundle.py
    python scripts/prepare_colab_bundle.py --out ~/Downloads/tj-bundle.zip

После сборки нужно:
  1. Загрузить zip в Google Drive → MyDrive/Colab Notebooks/tj-recs-bundle.zip
  2. Открыть в Colab notebooks/00_demo_serve_carousel.ipynb (либо тот же zip
     распаковать локально и открыть .ipynb через File → Upload).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_FILES = [
    # main pipeline scripts
    "serve_carousel.py",
    "train_ltr_full.py",
    "eval_ltr_i2i.py",
    "extra_analytics.py",
    "train_eval_no_coview_i2i.py",
    # доп. полезные утилиты, на которые могут ссылаться скрипты
    "build_recs_i2i.py",
    "make_example_serps.py",
    "recommender.py",
    "analyze_propensity.py",
    # demo
    "notebooks/00_demo_serve_carousel.ipynb",
    # meta
    "requirements.txt",
    "README.md",
    # каталог
    "tj_article.csv",
]

INCLUDE_DIRS = [
    "models",
    "coview_cache",
    "embeddings_cache",
]

# что не хотим тащить в Drive (даже если попало в INCLUDE_DIRS)
EXCLUDE_PATTERNS = (
    "__pycache__",
    ".DS_Store",
    "catboost_info",
)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n //= 1024
    return f"{n} TB"


def _iter_paths():
    for rel in INCLUDE_FILES:
        p = ROOT / rel
        if p.exists():
            yield p, rel
        else:
            print(f"  [warn] not found: {rel}")
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            print(f"  [warn] missing dir: {d}")
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PATTERNS for part in p.parts):
                continue
            rel = p.relative_to(ROOT).as_posix()
            yield p, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(ROOT / "tj-recs-bundle.zip"),
        help="куда сохранить zip‑бандл (по умолчанию: tj-recs-bundle.zip в корне)",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="только показать что войдёт в бандл, без архивирования",
    )
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()

    items = list(_iter_paths())
    total = sum(p.stat().st_size for p, _ in items)
    print(f"items: {len(items)}, raw size: {_human(total)}")
    print("-" * 70)
    for p, rel in items:
        print(f"  {_human(p.stat().st_size)}   {rel}")
    print("-" * 70)

    if args.list_only:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing zip: {out_path}")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p, rel in items:
            zf.write(p, arcname=f"tj-recs/{rel}")

    final = out_path.stat().st_size
    print(f"done. zip size: {_human(final)} (compression ratio {total / max(final, 1):.2f}x)")
    print()
    print("Дальше:")
    print(f"  1. Загрузить {out_path.name} в Google Drive")
    print("     (например, MyDrive/Colab Notebooks/tj-recs-bundle.zip)")
    print("  2. Открыть в Colab: notebooks/00_demo_serve_carousel.ipynb")
    print("     (его можно открыть прямо из распакованного zip)")
    print("  3. В первой ячейке указать BUNDLE_PATH = '/content/drive/MyDrive/...'")
    print("     и выполнить Runtime → Run all.")


if __name__ == "__main__":
    main()
