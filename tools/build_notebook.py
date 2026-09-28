"""Збирає notebooks/hw7_timeseries_jsonb.ipynb."""
import json
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
I = json.loads((ROOT / "tools/interpretations.json").read_text(encoding="utf-8"))

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))


def query(sql_text):
    code(f'q("""\n{sql_text.strip()}\n""")')


def ddl(sql_text, then=None):
    body = f'run("""\n{sql_text.strip()}\n""")'
    if then:
        body += f'\nq("""\n{then.strip()}\n""")'
    code(body)


def explain(label, sql_text):
    code(f'explain("{label}", """\n{sql_text.strip()}\n""")')


md("""
# ДЗ 7. Time-series ML features і JSONB event-log у PostgreSQL

**Студентка:** Samoilenko Mariia

- **Частина 1 — time-series:** реальні дані NYC TLC Yellow Taxi за **лютий і березень 2024** (варіант A, офіційні Parquet-файли),
  денна агрегація `t7_hw_taxi_daily` на рівні `(pickup_zone_id, ts_day)` і lag / rolling / expanding ознаки через window functions.
- **Частина 2 — JSONB:** синтетичний event-log `t7_hw_events` (50 000 подій, 5 типів), containment / key existence / JSONPath,
  GIN і expression B-tree індекси з `EXPLAIN (ANALYZE, BUFFERS)`, порівняння з нормалізованою таблицею.
""")

# ------------------------------------------------------------------ 1
md("## Завдання 1. Setup і завантаження даних")
code('''
import glob
import os
import subprocess
import sys


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


pip_install("psycopg2-binary", "sqlalchemy", "pandas", "pyarrow", "requests", "fasteners", "platformdirs", "psutil")

# pgserver 0.1.4 має wheel-файли лише до cp312; його розширення зібране в abi3,
# тому на новішому Python (Colab) встановлюємо cp312-wheel з тегом abi3.
if sys.version_info < (3, 13):
    pip_install("pgserver==0.1.4")
else:
    wheel_dir = "/tmp/pgserver_wheel"
    subprocess.check_call([
        sys.executable, "-m", "pip", "download", "-q", "pgserver==0.1.4",
        "--no-deps", "--only-binary=:all:", "--python-version", "3.12", "-d", wheel_dir,
    ])
    for whl in glob.glob(f"{wheel_dir}/pgserver-0.1.4-cp312-cp312-*.whl"):
        os.replace(whl, whl.replace("-cp312-cp312-", "-cp312-abi3-"))
    pip_install("--no-deps", *glob.glob(f"{wheel_dir}/pgserver-0.1.4-cp312-abi3-*.whl"))
''')
code('''
import io
import re
from urllib.request import urlretrieve

import pandas as pd
import pgserver
from sqlalchemy import create_engine, text

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120)

pg = pgserver.get_server("/tmp/hw7_pg", cleanup_mode="stop")
engine = create_engine(pg.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1), future=True)

with engine.connect() as conn:
    print(conn.execute(text("SELECT version()")).scalar_one())


def q(sql_text):
    """SELECT → DataFrame."""
    return pd.read_sql(text(sql_text), engine)


def run(sql_text):
    """DDL / DML в одній транзакції."""
    with engine.begin() as conn:
        conn.execute(text(sql_text))


PLANS = []


def explain(label, sql_text):
    """Виконати EXPLAIN (ANALYZE, BUFFERS), надрукувати план і зберегти ключові метрики."""
    with engine.connect() as conn:
        lines = [r[0] for r in conn.execute(text("EXPLAIN (ANALYZE, BUFFERS) " + sql_text))]
    plan = "\\n".join(lines)
    print(plan)
    nodes = [re.sub(r"\\s*\\(cost=.*", "", l).strip(" ->") for l in lines if "cost=" in l]
    cost = re.search(r"cost=[\\d.]+\\.\\.([\\d.]+)", lines[0])
    exec_ms = re.search(r"Execution Time: ([\\d.]+)", plan)
    PLANS.append({
        "step": label,
        "plan_nodes": " → ".join(nodes),
        "total_cost": float(cost.group(1)) if cost else None,
        "execution_ms": float(exec_ms.group(1)) if exec_ms else None,
    })
''')
md("""
### Варіант A: реальні дані NYC TLC

Відхилення від шаблону: `head(500_000)` одного березневого файлу покриває лише 1–6 березня 2024 (перевірено),
тому `trips_lag_7d` і `trips_lag_30d` майже завжди були б `NULL`, а комбінований view — порожнім. Беремо **повні** файли
за лютий і березень 2024 (≈ 6,6 млн поїздок, 60 днів, як у stub-схемі). Щоб завантаження мільйонів рядків було швидким,
замість `to_sql` рядки передаються в PostgreSQL через `COPY FROM STDIN` (таблиця створюється тим самим `to_sql` з порожнього DataFrame).
Фільтри очищення — як у шаблоні.
""")
code('''
DATA_DIR = "/content" if os.path.isdir("/content") else "/tmp"
MONTHS = ["2024-02", "2024-03"]
columns = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "passenger_count",
]

frames = []
for month in MONTHS:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
    path = os.path.join(DATA_DIR, f"yellow_tripdata_{month}.parquet")
    if not os.path.exists(path):
        urlretrieve(url, path)
    part = pd.read_parquet(path, columns=columns)
    print(month, "rows in file:", len(part))
    frames.append(part)
df = pd.concat(frames, ignore_index=True)

# Basic cleaning before loading to PostgreSQL
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]
df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]
df = df[df["PULocationID"].notna()]
print("rows after cleaning:", len(df))

run("DROP VIEW IF EXISTS v_t7_hw_taxi_features; DROP TABLE IF EXISTS nyc_taxi_raw CASCADE;")
df.head(0).to_sql("nyc_taxi_raw", engine, if_exists="replace", index=False)

raw_conn = engine.raw_connection()
try:
    with raw_conn.cursor() as cur:
        chunk = 1_000_000
        for start in range(0, len(df), chunk):
            buf = io.StringIO()
            df.iloc[start:start + chunk].to_csv(buf, index=False, header=False)
            buf.seek(0)
            cur.copy_expert(f"COPY nyc_taxi_raw ({', '.join(chr(34) + c + chr(34) for c in columns)}) FROM STDIN WITH (FORMAT csv)", buf)
    raw_conn.commit()
finally:
    raw_conn.close()
del frames, part

q("SELECT COUNT(*) AS n_rows FROM nyc_taxi_raw")
''')
md("""
### Daily aggregation `t7_hw_taxi_daily`

Додатково до фільтрів шаблону обмежуємо `ts_day` періодом **2024-02-01 … 2024-03-31**: у файлах TLC трапляються
поодинокі поїздки з датами на кшталт 2008-12-31 або кінця попереднього місяця, які створили б «дні» з 1–2 поїздками.
""")
ddl("""
DROP VIEW IF EXISTS v_t7_hw_taxi_features;
DROP TABLE IF EXISTS t7_hw_taxi_daily;

CREATE TABLE t7_hw_taxi_daily AS
SELECT
    "PULocationID"::INTEGER AS pickup_zone_id,
    DATE_TRUNC('day', tpep_pickup_datetime::TIMESTAMP)::DATE AS ts_day,
    COUNT(*)::INTEGER AS trip_count,
    AVG(fare_amount)::NUMERIC(10, 2) AS avg_fare,
    SUM(trip_distance)::NUMERIC(12, 2) AS total_distance
FROM nyc_taxi_raw
WHERE fare_amount > 0
  AND fare_amount < 500
  AND trip_distance > 0
  AND trip_distance < 100
  AND "PULocationID" IS NOT NULL
  AND tpep_pickup_datetime >= TIMESTAMP '2024-02-01'
  AND tpep_pickup_datetime <  TIMESTAMP '2024-04-01'
GROUP BY
    "PULocationID"::INTEGER,
    DATE_TRUNC('day', tpep_pickup_datetime::TIMESTAMP)::DATE;

ALTER TABLE t7_hw_taxi_daily
    ADD PRIMARY KEY (pickup_zone_id, ts_day);
""", then="""
SELECT pickup_zone_id, COUNT(*) AS n_days
FROM t7_hw_taxi_daily
GROUP BY pickup_zone_id
ORDER BY pickup_zone_id
LIMIT 10
""")
md("Загальний обсяг і повнота рядів:")
query("""
SELECT
    COUNT(*) AS n_rows,
    COUNT(DISTINCT pickup_zone_id) AS n_zones,
    MIN(ts_day) AS first_day,
    MAX(ts_day) AS last_day,
    SUM(trip_count) AS total_trips
FROM t7_hw_taxi_daily
""")
query("""
SELECT
    COUNT(*) AS zones,
    COUNT(*) FILTER (WHERE n_days = 60) AS complete_zones,
    COUNT(*) FILTER (WHERE n_days < 60) AS zones_with_gaps
FROM (SELECT pickup_zone_id, COUNT(*) AS n_days FROM t7_hw_taxi_daily GROUP BY pickup_zone_id) AS z
""")
md(I["daily"])
md("### Typed-таблиця з `TIMESTAMPTZ` (production-style момент часу)")
ddl("""
DROP TABLE IF EXISTS t7_hw_taxi_trips_typed;

CREATE TABLE t7_hw_taxi_trips_typed AS
SELECT
    "PULocationID"::INTEGER AS pickup_zone_id,
    "DOLocationID"::INTEGER AS dropoff_zone_id,
    (tpep_pickup_datetime::TIMESTAMP AT TIME ZONE 'America/New_York') AS pickup_at,
    (tpep_dropoff_datetime::TIMESTAMP AT TIME ZONE 'America/New_York') AS dropoff_at,
    trip_distance::NUMERIC(10, 2) AS trip_distance,
    fare_amount::NUMERIC(10, 2) AS fare_amount,
    passenger_count::INTEGER AS passenger_count
FROM nyc_taxi_raw
WHERE fare_amount > 0
  AND fare_amount < 500
  AND trip_distance > 0
  AND trip_distance < 100;
""", then="""
SELECT pickup_at, pg_typeof(pickup_at) AS pickup_type
FROM t7_hw_taxi_trips_typed
LIMIT 5
""")

# ------------------------------------------------------------------ 2
md(f"""
## Завдання 2. Time-series ML features

Усі ознаки рахуються з `PARTITION BY pickup_zone_id`. Для наочності приклади показано для зони **{I['zone']}** —
однієї з найзавантаженіших (вона має всі 60 днів). `WHERE` у зовнішньому запиті застосовується після обчислення вікон
у підзапиті, тому він не впливає на значення ознак.

### Feature 1. Lag features: lag-1d, lag-7d, lag-30d
""")
query(f"""
SELECT *
FROM (
    SELECT
        pickup_zone_id,
        ts_day,
        trip_count,
        LAG(trip_count, 1)  OVER w AS trips_lag_1d,
        LAG(trip_count, 7)  OVER w AS trips_lag_7d,
        LAG(trip_count, 30) OVER w AS trips_lag_30d
    FROM t7_hw_taxi_daily
    WINDOW w AS (
        PARTITION BY pickup_zone_id
        ORDER BY ts_day
    )
) AS f
WHERE pickup_zone_id = {I['zone']}
ORDER BY ts_day
LIMIT 20
""")
md(I["f1"])
md("### Feature 2. Rolling 7-day average fare (`RANGE ... INTERVAL '6 days' PRECEDING`)")
query(f"""
SELECT *
FROM (
    SELECT
        pickup_zone_id,
        ts_day,
        avg_fare,
        AVG(avg_fare) OVER w7 AS rolling_avg_fare_7d,
        COUNT(*)      OVER w7 AS days_in_window
    FROM t7_hw_taxi_daily
    WINDOW w7 AS (
        PARTITION BY pickup_zone_id
        ORDER BY ts_day::TIMESTAMP
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    )
) AS f
WHERE pickup_zone_id = {I['zone']}
ORDER BY ts_day
LIMIT 20
""")
md(I["f2"])
md("### Feature 3. Rolling 30-day fare volatility")
query(f"""
SELECT *
FROM (
    SELECT
        pickup_zone_id,
        ts_day,
        avg_fare,
        STDDEV_SAMP(avg_fare) OVER w30 AS rolling_std_fare_30d,
        COUNT(*)              OVER w30 AS days_in_window
    FROM t7_hw_taxi_daily
    WINDOW w30 AS (
        PARTITION BY pickup_zone_id
        ORDER BY ts_day::TIMESTAMP
        RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
    )
) AS f
WHERE pickup_zone_id = {I['zone']}
ORDER BY ts_day
LIMIT 20
""")
md(I["f3"])
md("### Feature 4. Expanding cumulative trips")
query(f"""
SELECT *
FROM (
    SELECT
        pickup_zone_id,
        ts_day,
        trip_count,
        SUM(trip_count) OVER we AS cumulative_trips,
        AVG(trip_count) OVER we AS expanding_avg_trips
    FROM t7_hw_taxi_daily
    WINDOW we AS (
        PARTITION BY pickup_zone_id
        ORDER BY ts_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
) AS f
WHERE pickup_zone_id = {I['zone']}
ORDER BY ts_day
LIMIT 20
""")
md(I["f4"])
md("""
### Combined feature view `v_t7_hw_taxi_features`

Усі чотири вікна (`w_pid`, `w7`, `w30`, `we`) містять `PARTITION BY pickup_zone_id`.

Додаткове покращення відносно шаблону: `LAG(trip_count, 7)` — це 7 **рядків** назад, а не 7 **днів**. У зонах із пропущеними
днями (коли не було жодної поїздки) рядок «7 назад» може бути 9 чи 12 днів тому. Тому view рахує ще й
`LAG(ts_day, k)` і залишає лише рядки, де всі lag-и точно відповідають 1, 7 і 30 календарним дням
(`lag_calendar_ok`). Rolling-ознаки з `RANGE ... INTERVAL` від пропусків не страждають: вікно визначається датами, а не рядками.
""")
ddl("""
DROP VIEW IF EXISTS v_t7_hw_taxi_features;

CREATE OR REPLACE VIEW v_t7_hw_taxi_features AS
WITH features_basic AS (
    SELECT
        pickup_zone_id,
        ts_day,
        trip_count,
        avg_fare,
        total_distance,

        LAG(trip_count, 1)  OVER w_pid AS trips_lag_1d,
        LAG(trip_count, 7)  OVER w_pid AS trips_lag_7d,
        LAG(trip_count, 30) OVER w_pid AS trips_lag_30d,

        (ts_day - LAG(ts_day, 1)  OVER w_pid = 1
         AND ts_day - LAG(ts_day, 7)  OVER w_pid = 7
         AND ts_day - LAG(ts_day, 30) OVER w_pid = 30) AS lag_calendar_ok,

        AVG(avg_fare) OVER w7 AS rolling_avg_fare_7d,
        STDDEV_SAMP(avg_fare) OVER w30 AS rolling_std_fare_30d,

        SUM(trip_count) OVER we AS cumulative_trips,
        AVG(trip_count) OVER we AS expanding_avg_trips
    FROM t7_hw_taxi_daily
    WINDOW
        w_pid AS (
            PARTITION BY pickup_zone_id
            ORDER BY ts_day
        ),
        w7 AS (
            PARTITION BY pickup_zone_id
            ORDER BY ts_day::TIMESTAMP
            RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
        ),
        w30 AS (
            PARTITION BY pickup_zone_id
            ORDER BY ts_day::TIMESTAMP
            RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
        ),
        we AS (
            PARTITION BY pickup_zone_id
            ORDER BY ts_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
)
SELECT *
FROM features_basic
WHERE trips_lag_30d IS NOT NULL
  AND lag_calendar_ok;
""", then=f"""
SELECT *
FROM v_t7_hw_taxi_features
WHERE pickup_zone_id = {I['zone']}
ORDER BY pickup_zone_id, ts_day
LIMIT 20
""")
query("""
SELECT
    COUNT(*) AS feature_rows,
    COUNT(DISTINCT pickup_zone_id) AS zones,
    MIN(ts_day) AS first_day,
    MAX(ts_day) AS last_day
FROM v_t7_hw_taxi_features
""")
md("**Чому `PARTITION BY` критичний** — порівнюємо LAG з партиціюванням і без нього на першому дні кожної зони:")
query("""
WITH lags AS (
    SELECT
        pickup_zone_id,
        ts_day,
        LAG(trip_count, 1) OVER (PARTITION BY pickup_zone_id ORDER BY ts_day) AS lag_1d_partitioned,
        LAG(trip_count, 1) OVER (ORDER BY pickup_zone_id, ts_day)            AS lag_1d_no_partition,
        ROW_NUMBER()       OVER (PARTITION BY pickup_zone_id ORDER BY ts_day) AS rn
    FROM t7_hw_taxi_daily
)
SELECT
    COUNT(*) FILTER (WHERE rn = 1)                                          AS first_rows_of_zones,
    COUNT(*) FILTER (WHERE rn = 1 AND lag_1d_partitioned IS NULL)           AS partitioned_null_as_expected,
    COUNT(*) FILTER (WHERE rn = 1 AND lag_1d_no_partition IS NOT NULL)      AS no_partition_leaked_from_other_zone
FROM lags
""")
md(I["partition"])

# ------------------------------------------------------------------ 3
md("## Завдання 3. JSONB-таблиця для подій")
ddl("""
DROP TABLE IF EXISTS t7_hw_checkouts_norm;
DROP TABLE IF EXISTS t7_hw_events;

CREATE TABLE t7_hw_events (
    event_id    INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    event_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL
);

INSERT INTO t7_hw_events (user_id, event_ts, event_type, payload)
SELECT
    (n % 1000) + 1 AS user_id,
    TIMESTAMPTZ '2024-04-01 00:00:00+00' - ((n % 60) || ' days')::INTERVAL AS event_ts,
    CASE (n % 5)
        WHEN 0 THEN 'page_view'
        WHEN 1 THEN 'click'
        WHEN 2 THEN 'add_to_cart'
        WHEN 3 THEN 'checkout'
        ELSE        'recommendation_shown'
    END AS event_type,
    CASE (n % 5)
        WHEN 0 THEN jsonb_build_object(
            'url', '/product/' || ((n % 200) + 1),
            'referrer', (ARRAY['google','direct','email','fb'])[(n % 4) + 1],
            'device', (ARRAY['mobile','desktop','tablet'])[(n % 3) + 1]
        )
        WHEN 1 THEN jsonb_build_object(
            'element_id', 'btn_' || ((n % 50) + 1),
            'page_url', '/category/' || ((n % 20) + 1),
            'device', (ARRAY['mobile','desktop'])[(n % 2) + 1]
        )
        WHEN 2 THEN jsonb_build_object(
            'product_id', (n % 200) + 1,
            'price', ((n % 100) + 10)::NUMERIC(10, 2),
            'quantity', (n % 5) + 1,
            'tags', jsonb_build_array((ARRAY['promo','new','clearance'])[(n % 3) + 1])
        )
        WHEN 3 THEN jsonb_build_object(
            'order_id', 'o_' || (n + 100000),
            'total', ((n % 500) + 50)::NUMERIC(10, 2),
            'items_count', (n % 10) + 1,
            'payment_method', (ARRAY['card','paypal','crypto'])[(n % 3) + 1]
        )
        ELSE jsonb_build_object(
            'model_name', 'rec_v' || ((n % 5) + 1),
            'model_version', '1.' || (n % 10),
            'recommendations', jsonb_build_array((n % 200) + 1, ((n + 1) % 200) + 1, ((n + 2) % 200) + 1),
            'experiment', 'exp_' || ((n % 3) + 1)
        )
    END AS payload
FROM generate_series(1, 50000) AS n;

ANALYZE t7_hw_events;
""", then="""
SELECT event_type, COUNT(DISTINCT event_id) AS n_events,
       (ARRAY_AGG(DISTINCT k ORDER BY k))::TEXT AS payload_keys
FROM t7_hw_events, LATERAL jsonb_object_keys(payload) AS k
GROUP BY event_type
ORDER BY event_type
""")
md(I["events"])
md("### JSONB Query 1. Containment `@>`")
query("""
SELECT event_id, user_id, event_ts, payload
FROM t7_hw_events
WHERE event_type = 'checkout'
  AND payload @> jsonb_build_object('payment_method', 'card')
LIMIT 10
""")
md("### JSONB Query 2. Key existence `?` та `?|`")
query("""
SELECT event_id, event_type, payload
FROM t7_hw_events
WHERE payload ? 'tags'
LIMIT 10
""")
query("""
SELECT event_id, event_type, payload
FROM t7_hw_events
WHERE payload ?| ARRAY['experiment', 'model_version']
LIMIT 10
""")
md("### JSONB Query 3. JSONPath")
query("""
SELECT
    event_id,
    jsonb_path_query(payload, '$.recommendations[*]') AS recommended_pid
FROM t7_hw_events
WHERE event_type = 'recommendation_shown'
LIMIT 15
""")
query("""
SELECT
    event_id,
    payload ->> 'order_id' AS order_id,
    (payload ->> 'total')::NUMERIC(10, 2) AS total
FROM t7_hw_events
WHERE event_type = 'checkout'
  AND jsonb_path_exists(payload, '$.total ? (@ > 300)')
LIMIT 10
""")
md("Скільки рядків повертає кожен тип фільтра на всій таблиці (для оцінки селективності в Завданні 4):")
query("""
SELECT
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE payload @> '{"payment_method": "card"}') AS card_checkouts,
    COUNT(*) FILTER (WHERE payload @> '{"referrer": "google"}')     AS google_page_views,
    COUNT(*) FILTER (WHERE payload ? 'tags')                         AS with_tags,
    COUNT(*) FILTER (WHERE payload ?| ARRAY['experiment', 'model_version']) AS rec_events,
    COUNT(*) FILTER (WHERE jsonb_path_exists(payload, '$.total ? (@ > 300)')) AS checkouts_total_over_300,
    COUNT(*) FILTER (WHERE payload ->> 'device' = 'mobile')          AS mobile_events
FROM t7_hw_events
""")
md(I["jsonb"])

# ------------------------------------------------------------------ 4
md("## Завдання 4. GIN-index demo + `EXPLAIN (ANALYZE, BUFFERS)`\n\n### Step 1. Containment query без GIN-індексу")
ddl("""
DROP INDEX IF EXISTS idx_t7_events_payload_gin_pathops;
DROP INDEX IF EXISTS idx_t7_events_payload_gin_ops;
DROP INDEX IF EXISTS idx_t7_events_device_btree;

ANALYZE t7_hw_events;
""")
explain("1. @> card — без GIN", """
SELECT COUNT(*)
FROM t7_hw_events
WHERE payload @> jsonb_build_object('payment_method', 'card')
""")
md("### Step 2. GIN-індекс `jsonb_path_ops`")
ddl("""
CREATE INDEX idx_t7_events_payload_gin_pathops
    ON t7_hw_events USING GIN (payload jsonb_path_ops);

ANALYZE t7_hw_events;
""", then="""
SELECT
    pg_size_pretty(pg_relation_size('idx_t7_events_payload_gin_pathops')) AS gin_index_size,
    pg_size_pretty(pg_relation_size('t7_hw_events'))                      AS table_size
""")
md("### Step 3. Той самий containment query після GIN")
explain("2. @> card — після GIN", """
SELECT COUNT(*)
FROM t7_hw_events
WHERE payload @> jsonb_build_object('payment_method', 'card')
""")
explain("3. @> google — після GIN", """
SELECT COUNT(*)
FROM t7_hw_events
WHERE payload @> jsonb_build_object('referrer', 'google')
""")
md("### Step 4. Коли GIN не допомагає: path-specific text equality")
explain("4. ->> device = mobile — лише GIN", """
SELECT COUNT(*)
FROM t7_hw_events
WHERE payload ->> 'device' = 'mobile'
""")
ddl("""
CREATE INDEX idx_t7_events_device_btree
    ON t7_hw_events ((payload ->> 'device'));

ANALYZE t7_hw_events;
""")
explain("5. ->> device = mobile — expression B-tree", """
SELECT COUNT(*)
FROM t7_hw_events
WHERE payload ->> 'device' = 'mobile'
""")
md("### Зведення планів")
code("pd.DataFrame(PLANS)")
md(I["explain"])

# ------------------------------------------------------------------ 5
md("## Завдання 5. JSONB vs normalized\n\n### Step 1. Нормалізована таблиця для checkout-подій")
ddl("""
DROP TABLE IF EXISTS t7_hw_checkouts_norm;

CREATE TABLE t7_hw_checkouts_norm (
    event_id        INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    event_ts        TIMESTAMPTZ NOT NULL,
    order_id        TEXT NOT NULL,
    total           NUMERIC(10, 2) NOT NULL,
    items_count     INTEGER NOT NULL CHECK (items_count > 0),
    payment_method  TEXT NOT NULL CHECK (payment_method IN ('card', 'paypal', 'crypto'))
);

INSERT INTO t7_hw_checkouts_norm (event_id, user_id, event_ts, order_id, total, items_count, payment_method)
SELECT
    event_id,
    user_id,
    event_ts,
    payload ->> 'order_id' AS order_id,
    (payload ->> 'total')::NUMERIC(10, 2) AS total,
    (payload ->> 'items_count')::INTEGER AS items_count,
    payload ->> 'payment_method' AS payment_method
FROM t7_hw_events
WHERE event_type = 'checkout';

CREATE INDEX idx_t7_checkouts_norm_pm
    ON t7_hw_checkouts_norm (payment_method);

ANALYZE t7_hw_checkouts_norm;
""", then="""
SELECT COUNT(*) AS n_checkouts FROM t7_hw_checkouts_norm
""")
md("### Step 2. Один запит двома способами\n\n**Результат (однаковий для обох версій):**")
query("""
SELECT payment_method, ROUND(AVG(total), 2) AS avg_total, COUNT(*) AS n
FROM t7_hw_checkouts_norm
GROUP BY payment_method
ORDER BY avg_total DESC
LIMIT 3
""")
md("**JSONB-версія:**")
explain("6. checkout avg — JSONB", """
SELECT
    payload ->> 'payment_method' AS payment_method,
    AVG((payload ->> 'total')::NUMERIC) AS avg_total,
    COUNT(*) AS n
FROM t7_hw_events
WHERE event_type = 'checkout'
GROUP BY payload ->> 'payment_method'
ORDER BY avg_total DESC
LIMIT 3
""")
md("**Normalized-версія:**")
explain("7. checkout avg — normalized", """
SELECT
    payment_method,
    AVG(total) AS avg_total,
    COUNT(*) AS n
FROM t7_hw_checkouts_norm
GROUP BY payment_method
ORDER BY avg_total DESC
LIMIT 3
""")
code("pd.DataFrame(PLANS[-2:])")
md(I["norm"])
md("""
### Step 3. Bonus: MongoDB-equivalent (pseudocode, не виконується)

```javascript
// MongoDB equivalent: avg total checkout per payment_method, top-3
// Pseudocode — не виконується у PostgreSQL

db.events.aggregate([
  {$match: {event_type: "checkout"}},
  {$group: {
    _id: "$payload.payment_method",
    avg_total: {$avg: {$toDouble: "$payload.total"}},
    n: {$sum: 1}
  }},
  {$sort: {avg_total: -1}},
  {$limit: 3}
]);
```

- **PostgreSQL JSONB** зручний, коли поруч потрібні SQL, JOIN з нормалізованими таблицями (users, orders),
  транзакції й constraints: подію та зміну пов'язаних таблиць можна записати атомарно.
- **MongoDB aggregation pipeline** природний для document-first навантажень: `$match → $group → $sort → $limit`
  описує той самий запит як послідовність стадій над документами без фіксованої схеми.
- **High-volume raw clickstream** (мільярди подій, append-only, рідкісні точкові читання) дешевше тримати в document store
  або data lake (Parquet в об'єктному сховищі), а не в OLTP-базі з GIN-індексами.
- **Curated ML features** зручніше тримати в типізованих SQL-таблицях: типи, `CHECK`, B-tree індекси, простий JOIN
  з мітками й стабільна схема для training / serving.
""")
md("## Reflection\n\n" + I["reflection"])

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, ROOT / "notebooks/hw7_timeseries_jsonb.ipynb")
print("wrote notebook")
