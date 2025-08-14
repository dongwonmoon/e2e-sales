BEGIN;

-- Cutoff
DROP TABLE IF EXISTS _cutoff;
CREATE TEMP TABLE _cutoff AS
WITH days AS (
    SELECT DISTINCT date::date AS d FROM features
),
cut AS (
    SELECT percentile_disc(0.3) WITHIN GROUP (ORDER BY d) AS cutoff_date
    FROM days
)
SELECT cutoff_date FROM cut;

-- Train
DROP TABLE IF EXISTS sales_train;
CREATE TABLE sales_train (LIKE sales INCLUDING ALL);

DROP TABLE IF EXISTS features_train;
CREATE TABLE features_train (LIKE features INCLUDING ALL);

INSERT INTO sales_train
SELECT *
FROM sales
WHERE date::date <= (SELECT cutoff_date FROM _cutoff);

INSERT INTO features_train
SELECT *
FROM features
WHERE date::date <= (SELECT cutoff_date FROM _cutoff);

-- Stream
DROP TABLE IF EXISTS sales_stream;
CREATE TABLE sales_stream (LIKE sales INCLUDING ALL);

DROP TABLE IF EXISTS features_stream;
CREATE TABLE features_stream (LIKE features INCLUDING ALL);

INSERT INTO sales_stream
SELECT *
FROM sales
WHERE date::date > (SELECT cutoff_date FROM _cutoff);

INSERT INTO features_stream
SELECT *
FROM features
WHERE date::date > (SELECT cutoff_date FROM _cutoff);

-- State
DROP TABLE IF EXISTS state;
CREATE TABLE state (
  id SMALLINT PRIMARY KEY,
  cur_time TIMESTAMP NOT NULL
);

INSERT INTO state (id, cur_time)
SELECT
  1,
  (MIN(date)::timestamp - INTERVAL '1 day')
FROM features_stream
ON CONFLICT (id) DO UPDATE
SET cur_time = EXCLUDED.cur_time;

COMMIT;
