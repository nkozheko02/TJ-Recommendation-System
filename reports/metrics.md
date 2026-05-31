# Метрики и результаты оффлайн‑анализа

Дата обновления: 2026-04-29

## Контекст

- **Логи**: `tj_session_w_target_full.parquet` (использовали сэмплы; `source_type="coview"` исключали из показов при оценке LTR).
- **Сплит по времени**: train < `2026-03-20` (UTC), test ≥ `2026-03-20` (UTC).
- **Метрики ранжирования**: nDCG@6, MRR@6 (query = `impression_key`).
- **Debias**: propensity-weighting по позиции w = 1 / \max(p(click|pos), 0.02), propensity считали на train.

---

## Retrieval: Recall@N (попадает ли клик в candidate pool)

Считали на 5000 кликающих показах (target=1), источники выравнены с эмбеддингами.

Recall@N

### Recall@N

- **@50**: embeddings **0.5116**, coview **0.5622**, union **0.7504**
- **@100**: embeddings **0.5742**, coview **0.6040**, union **0.7948**
- **@200**: embeddings **0.6326**, coview **0.6278**, union **0.8264**
- **@400**: embeddings **0.6960**, coview **0.6278**, union **0.8450**
- **@800**: embeddings **0.7418**, coview **0.6278**, union **0.8606**

Вывод: **union (embeddings + coview)** резко повышает покрытие кликов в candidate pool.

---

## LTR: time-split (общие метрики)

Сэмпл логов показов (без `source_type="coview"`), train_imps=9903, test_imps=20000.

LTR overall

### Без debias (веса = 1)

- **nDCG@6**: LightGBM **0.02004**, CatBoost **0.01998**
- **MRR@6**: LightGBM **0.01875**, CatBoost **0.01868**

### С debias по позиции (propensity-weighting)

- **nDCG@6**: LightGBM **0.02017**, CatBoost **0.02002**
- **MRR@6**: LightGBM **0.01891**, CatBoost **0.01874**

Вывод: на текущих фичах **LightGBM стабильно чуть лучше**; debias даёт **небольшой положительный сдвиг**.

---

## Propensity по позиции (train)

Оценка p(click|pos) со сглаживанием Beta(1,1):

Propensity by position

- `pos=0`: **0.75**
- `pos=1`: **0.0168**
- `pos=2`: **0.0158**
- `pos=3`: **0.0151**
- `pos=4`: **0.0361**
- `pos=5`: **0.0167**

---

## LTR: метрики по `entity_type` (test, с debias)

Показаны топ‑типы по числу импрессий.

- `**article.ml_personal-block-recommendation`** (n=11434)
  - nDCG@6: LGBM **0.01514**, CatBoost **0.01493**
  - MRR@6: LGBM **0.01447**, CatBoost **0.01421**
- `**article.ml_what-else-mi-pisali-block-recommendation`** (n=4251)
  - nDCG@6: LGBM **0.04544**, CatBoost **0.04544**
  - MRR@6: LGBM **0.04209**, CatBoost **0.04219**
- `**article.popularity-block-recommendation`** (n=3587)
  - nDCG@6: LGBM **0.00571**, CatBoost **0.00532**
  - MRR@6: LGBM **0.00520**, CatBoost **0.00469**
- `**article.what-else-mi-pisali-block-recommendation`** (n=728)
  - nDCG@6: LGBM **0.02291**, CatBoost **0.02392**
  - MRR@6: LGBM **0.02083**, CatBoost **0.02221**

Примечание: на малых n возможны флуктуации, поэтому важнее смотреть типы с большим числом импрессий.

---

## Feature importance (LightGBM, train, с debias)

Тип importance: **gain** (доля суммарного gain по всем деревьям) и **split** (сколько раз фича использовалась в разбиениях).

Feature importance

Топ‑фичи по gain:

- **sim**: gain_share=0.146, split=2268
- **cand_log_comments**: gain_share=0.120, split=1783
- **cand_like_rate**: gain_share=0.103, split=1385
- **abs_age_diff_days**: gain_share=0.096, split=1759
- **cand_fresh**: gain_share=0.090, split=1164
- **cand_fav_rate**: gain_share=0.074, split=1415
- **cand_comment_rate**: gain_share=0.071, split=1157
- **cand_log_views**: gain_share=0.064, split=1461
- **src_like_rate**: gain_share=0.063, split=877
- **src_fresh**: gain_share=0.051, split=965
- **src_log_views**: gain_share=0.050, split=899
- **pos**: gain_share=0.037, split=587
- **same_rubric**: gain_share=0.026, split=270
- **same_dept**: gain_share=0.006, split=89
- **same_author**: gain_share=0.004, split=103

Короткая интерпретация:

- `sim` — главный сигнал (ожидаемо).
- качество/вовлечённость кандидата (`like_rate`, `log_comments`, `log_views`, `fav_rate`) и контекст источника (`src_fresh`, `src_like_rate`) дают сопоставимый вклад.
- парная фича `abs_age_diff_days` стала **топ‑4** по gain → полезна для “серийности/актуальности” и удержания темы в адекватных временных рамках.
- `pos` остаётся важным даже при propensity-weighting → bias полностью не «исчезает», но его влияние снижается.

