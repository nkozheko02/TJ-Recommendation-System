# Google Colab — как запустить ноутбуки

В этой папке лежат `.ipynb`-копии всех ключевых скриптов проекта.
Они построены так, чтобы:

* **листать как презентацию** — каждая функция/класс в отдельной ячейке, слева в Colab доступна навигация через **Table of contents** (иконка списка);
* **прогонять код** прямо в Colab — для этого первая ячейка каждого ноутбука настроит окружение.

## Один раз перед демо

1. Собери архив с артефактами **локально на маке**:

   ```bash
   python scripts/build_colab_bundle.py            # ~440 MB — для serve_carousel и аналитики
   python scripts/build_colab_bundle.py --with-logs  # +2.2 GB лог — для train/eval
   ```

   На выходе получаются `notebooks/tj_recs_colab_inference.zip` и (опционально) `notebooks/tj_recs_colab_train.zip`.

2. Залей в свой **Google Drive** (`My Drive/` — то есть корень):

   * `tj_recs_colab_inference.zip` — **обязательно**;
   * `tj_recs_colab_train.zip` *или* отдельно файл `tj_session_w_target_full.parquet` — только если хочешь реально обучать/мерить модель в Colab. Для презентации это не нужно — метрики уже сохранены в `reports/*.json`.

3. Залей в Drive сами ноутбуки (всю папку `notebooks/`).

## На самой защите

1. Открой Colab, дальше `File → Open notebook → Google Drive → notebooks/serve_carousel.ipynb`.
2. Выполни первую ячейку **Colab Bootstrap** — она:
   * смонтирует Drive (Google спросит подтверждение);
   * распакует bundle в `/content/tj-recs`;
   * сделает `chdir` и добавит путь в `sys.path`;
   * поставит `lightgbm`, `catboost`, `pyarrow`.
3. Дальше можно либо последовательно выполнять ячейки, либо листать как презентацию через **Table of contents**.
4. В `serve_carousel.ipynb` в самом конце есть **готовая demo-ячейка** — она строит карусель для одной статьи прямо в ноутбуке (`ctx = build_context()` → `serve_one(ctx, article_id)`).

## Какой ноутбук что показывает

| Файл | Что показывает | Можно ли реально прогнать в Colab |
|---|---|---|
| `serve_carousel.ipynb` | Inference — retrieval + LTR + reranking, выдача карусели | ✅ да, есть demo-ячейка с готовым вызовом |
| `train_ltr_full.ipynb` | Полный пайплайн обучения LTR | ⚠ только если залит `tj_session_w_target_full.parquet` (2.2 GB) |
| `eval_ltr_i2i.ipynb` | Оффлайн-оценка модели только на i2i | ⚠ нужен parquet, иначе только просмотр |
| `extra_analytics.ipynb` | Дополнительные графики и метрики | ⚠ нужен parquet для пересчёта; без него — листать как презентацию |
| `train_eval_no_coview_i2i.ipynb` | A/B-сравнение LTR с coview и без | ⚠ нужен parquet |

## Если что-то пошло не так

* **«не нашёл `tj_recs_colab_inference.zip`»** — проверь, что архив лежит в **корне** `My Drive`, а не во вложенной папке. Если другая папка — поправь `BUNDLE_PATH` в bootstrap-ячейке.
* **`pip install` падает** — выполни ячейку ещё раз; первая попытка иногда таймаутит на скачивании катбуста.
* **`ModuleNotFoundError: train_ltr_full`** — значит bootstrap не отработал. Проверь, что `os.getcwd()` указывает на `/content/tj-recs` и что там есть `train_ltr_full.py`.
* **«too long» при распаковке** — bundle разворачивается ~1-2 мин. Если запустить ту же ячейку повторно, она пропустит распаковку.
