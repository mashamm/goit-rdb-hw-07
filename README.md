# goit-rdb-hw-07 — Time-series ML features і JSONB event-log у PostgreSQL

**Автор:** Samoilenko Mariia

## Джерело даних

- **Time-series:** NYC TLC Trip Record Data — Yellow Taxi, **лютий і березень 2024** (варіант A, реальні дані, stub-schema не використовується).
  - Офіційна сторінка: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  - Файли: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet` і `..._2024-03.parquet`
- **JSONB event-log:** синтетична таблиця `t7_hw_events` (50 000 подій, генерується SQL через `generate_series`).

## Спосіб отримання файлів

Notebook сам завантажує обидва Parquet-файли з офіційного URL (у репозиторії їх немає), залишає 7 колонок, застосовує фільтри очищення
(`0 < fare_amount < 500`, `0 < trip_distance < 100`, `PULocationID IS NOT NULL`) і завантажує 6 341 321 поїздку в `nyc_taxi_raw`
через `COPY FROM STDIN`. Два місяці замість `head(500_000)` одного місяця взято тому, що перші 500 000 рядків березневого файлу
покривають лише 1–6 березня, і lag-7d / lag-30d були б майже завжди `NULL`.

## Time-series features (`t7_hw_taxi_daily`, `v_t7_hw_taxi_features`)

- Денна агрегація на рівні `(pickup_zone_id, ts_day)`: 13 511 рядків, 258 зон, 2024-02-01 … 2024-03-31.
- Усі вікна мають `PARTITION BY pickup_zone_id`:
  - lag: `trips_lag_1d`, `trips_lag_7d`, `trips_lag_30d` (+ перевірка, що lag відповідає рівно 1/7/30 календарним дням, бо в частини зон є дні без поїздок);
  - rolling 7-day average fare: `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW`;
  - rolling 30-day fare volatility: `STDDEV_SAMP` за `INTERVAL '29 days'`;
  - expanding: `cumulative_trips`, `expanding_avg_trips` (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`).
- Демонстрація, чому без `PARTITION BY` LAG «перетікає» між зонами.

## JSONB-частина

- `t7_hw_events`: 5 типів подій (`page_view`, `click`, `add_to_cart`, `checkout`, `recommendation_shown`) з різними схемами payload.
- Запити: containment `@>`, key existence `?` / `?|`, JSONPath (`jsonb_path_query`, `jsonb_path_exists`).
- `EXPLAIN (ANALYZE, BUFFERS)` до/після `GIN (payload jsonb_path_ops)` і до/після expression B-tree `((payload ->> 'device'))`.
- Порівняння JSONB-запиту з нормалізованою `t7_hw_checkouts_norm`, bonus: MongoDB-equivalent (pseudocode).

## Запуск

Відкрийте `notebooks/hw7_timeseries_jsonb.ipynb` у Google Colab і виконайте **Runtime → Restart session and run all**
(≈ 2–3 хв: завантаження ~105 МБ Parquet і COPY 6,3 млн рядків). PostgreSQL 16 запускається всередині notebook через `pgserver`.
