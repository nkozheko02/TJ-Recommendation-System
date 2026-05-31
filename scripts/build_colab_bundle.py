"""Собирает zip-архив с минимальным набором артефактов для запуска ноутбуков
в Google Colab.

Делает два бандла:

1. **inference** (`tj_recs_colab_inference.zip`, ~550 MB) — всё что нужно для
   `serve_carousel.ipynb` и `extra_analytics.ipynb` (последний работает без
   parquet-логов, если задать `--mixed` не использовать — он строит метрики
   только из артефактов; для full mode нужен parquet, см. ниже).

2. **train** (`tj_recs_colab_train.zip`, опционально) — добавляет к inference
   тяжёлый исходный лог `tj_session_w_target_full.parquet` (~2.2 GB) для
   `train_ltr_full.ipynb`, `eval_ltr_i2i.ipynb`,
   `train_eval_no_coview_i2i.ipynb`. Включается флагом `--with-logs`.

Использование:
    python scripts/build_colab_bundle.py              # inference bundle
    python scripts/build_colab_bundle.py --with-logs  # + лог-parquet
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks"
LOGS_PARQUET = Path("/Users/nkozheko/Downloads/tj_session_w_target_full.parquet")

# Файлы, нужные для inference и большинства аналитических ноутбуков.
INFERENCE_FILES: list[tuple[Path, str]] = [
    # каталог статей
    (ROOT / "tj_article.csv", "tj_article.csv"),
    # эмбеддинги (memmap)
    (ROOT / "embeddings_cache" / "embeddings.f32", "embeddings_cache/embeddings.f32"),
    (ROOT / "embeddings_cache" / "ids.txt", "embeddings_cache/ids.txt"),
    (ROOT / "embeddings_cache" / "shape.txt", "embeddings_cache/shape.txt"),
    # coview-индекс
    (ROOT / "coview_cache" / "coview_index.pkl.gz", "coview_cache/coview_index.pkl.gz"),
    # модели
    (ROOT / "models" / "ltr_lgbm.pkl", "models/ltr_lgbm.pkl"),
    (ROOT / "models" / "ltr_catboost.cbm", "models/ltr_catboost.cbm"),
    (ROOT / "models" / "ltr_lgbm_no_coview.pkl", "models/ltr_lgbm_no_coview.pkl"),
    # .py-файлы с исходным кодом (нужны для импорта из ноутбуков)
    (ROOT / "serve_carousel.py", "serve_carousel.py"),
    (ROOT / "train_ltr_full.py", "train_ltr_full.py"),
    (ROOT / "eval_ltr_i2i.py", "eval_ltr_i2i.py"),
    (ROOT / "extra_analytics.py", "extra_analytics.py"),
    (ROOT / "train_eval_no_coview_i2i.py", "train_eval_no_coview_i2i.py"),
    # уже посчитанные метрики/отчёты — полезны если parquet недоступен
    (ROOT / "reports" / "ltr_full_metrics.json", "reports/ltr_full_metrics.json"),
    (ROOT / "reports" / "ltr_full_metrics_i2i.json", "reports/ltr_full_metrics_i2i.json"),
    (ROOT / "reports" / "ltr_extra_metrics.json", "reports/ltr_extra_metrics.json"),
    (ROOT / "reports" / "ltr_extra_metrics_i2i.json", "reports/ltr_extra_metrics_i2i.json"),
    (ROOT / "reports" / "ltr_extra_metrics_no_coview_i2i.json", "reports/ltr_extra_metrics_no_coview_i2i.json"),
    (ROOT / "reports" / "coview_uplift_i2i.json", "reports/coview_uplift_i2i.json"),
    (ROOT / "reports" / "propensity_by_entity.csv", "reports/propensity_by_entity.csv"),
    (ROOT / "reports" / "ltr_full_per_entity.csv", "reports/ltr_full_per_entity.csv"),
    (ROOT / "reports" / "ltr_extra_per_entity.csv", "reports/ltr_extra_per_entity.csv"),
    (ROOT / "reports" / "ltr_extra_per_entity_i2i.csv", "reports/ltr_extra_per_entity_i2i.csv"),
    # пример рекомендаций (на случай если захотим показать выходную таблицу)
    (ROOT / "recs_i2i_sample.csv", "recs_i2i_sample.csv"),
]


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def build_zip(out_path: Path, files: list[tuple[Path, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    missing: list[str] = []
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for src, arcname in files:
            if not src.exists():
                missing.append(str(src))
                continue
            size = src.stat().st_size
            total += size
            print(f"  + {arcname:<55}  {_fmt_size(size):>10}")
            zf.write(src, arcname=arcname)

    if missing:
        print("\n[warn] не найдены файлы:")
        for m in missing:
            print(f"  - {m}")

    print(f"\nИтого: {out_path.name}  ({_fmt_size(out_path.stat().st_size)} архив, {_fmt_size(total)} сырых)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-logs", action="store_true", help="добавить parquet-лог (~2.2 GB) для train/eval ноутбуков")
    ap.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="куда положить zip (по умолчанию notebooks/)",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    print("== Inference bundle ==")
    build_zip(out_dir / "tj_recs_colab_inference.zip", INFERENCE_FILES)

    if args.with_logs:
        print("\n== Train bundle ==")
        train_files = list(INFERENCE_FILES)
        train_files.append((LOGS_PARQUET, "tj_session_w_target_full.parquet"))
        build_zip(out_dir / "tj_recs_colab_train.zip", train_files)


if __name__ == "__main__":
    main()
