DROP TABLE IF EXISTS sales;
CREATE TABLE sales (
    store INT,
    dept INT,
    date TIMESTAMP,
    weekly_sales FLOAT,
    is_holiday BOOLEAN
);

COPY sales FROM '/tmp/data/raw/train.csv' DELIMITER ',' CSV HEADER NULL 'NA'; 

-- CSV 원본 컬럼 전부 받는 임시/스테이징 테이블
CREATE TABLE features_stg (
  store INT,
  date  TIMESTAMP,
  temperature FLOAT,
  fuel_price FLOAT,
  markdown1 FLOAT,
  markdown2 FLOAT,
  markdown3 FLOAT,
  markdown4 FLOAT,
  markdown5 FLOAT,
  cpi FLOAT,
  unemployment FLOAT,
  isholiday TEXT
);

-- 원본 CSV 전체를 먼저 적재
COPY features_stg FROM '/tmp/data/raw/features.csv' DELIMITER ',' CSV HEADER NULL 'NA';

CREATE TABLE features (
    store INT,
    date TIMESTAMP,
    temperature FLOAT,
    fuel_price FLOAT,
    cpi FLOAT,
    unemployment FLOAT,
    is_holiday BOOLEAN
);

-- 필요한 컬럼만 선별해서 타입 캐스팅 후 본 테이블로 적재
INSERT INTO features (store, date, temperature, fuel_price, cpi, unemployment, is_holiday)
SELECT
  store,
  date,
  temperature,
  fuel_price,
  cpi,
  unemployment,
  CASE LOWER(isholiday)
    WHEN 'true' THEN TRUE
    WHEN 'false' THEN FALSE
    ELSE NULL
  END AS is_holiday
FROM features_stg;

-- 다 적재 후 정리
DROP TABLE features_stg;