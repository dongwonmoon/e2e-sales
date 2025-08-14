from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowException
import pendulum
import os
from typing import Optional
import pandas as pd
import json
from pathlib import Path
import logging
import time
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.pyfunc
from common_lib.model_wrapper import SklearnModelWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _t0():
    return time.perf_counter()


def _elapsed(s):
    return f"{time.perf_counter() - s:.3f}s"


# ========== Config ==========
class Config:
    # Basic
    project_db_host = os.environ.get("PROJECT_DB_HOST", "project-db")
    project_db_port = os.environ.get("PROJECT_DB_PORT", 5432)
    project_db_user = os.environ.get("PROJECT_DB_USER", "sales")
    project_db_password = os.environ.get("PROJECT_DB_PASSWORD", "sales")
    project_db_name = os.environ.get("PROJECT_DB_NAME", "sales")
    project_db_dsn = f"postgresql://{project_db_user}:{project_db_password}@{project_db_host}:{project_db_port}/{project_db_name}"

    # Data & Model Directories
    data_dir: str = "/opt/airflow/data"
    models_dir: str = "/opt/airflow/models"

    # Model
    model_params: dict = {"n_estimators": 300, "n_jobs": 1, "random_state": 42}
    mlflow_experiment_name: str = "sales_daily"


# ========= Airflow DAG 정의 =========
@dag(
    dag_id="daily_train_sales",
    schedule="10 0 * * *",  # 매일 00:10 (UTC)에 실행
    start_date=pendulum.datetime(2025, 8, 11, tz=pendulum.timezone("Asia/Seoul")),
    catchup=False,
    max_active_runs=1,
    tags=["sales", "mlops"],
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": pendulum.duration(minutes=5),
    }
)
def daily_train_sales():
    """
    매일 판매 데이터를 학습하여 모델을 업데이트하는 DAG.
    - Step 1: `compute_cutoff`: 논리적 실행 시간을 기준으로 데이터 추출 마감 시각을 계산합니다.
    - Step 2: `load_data`: DB에서 해당 시점까지의 모든 데이터를 CSV 파일로 저장합니다.
    - Step 3: `train`: CSV 데이터를 읽어 모델을 학습하고 MLflow에 기록합니다.
    - Step 4: `validate`: 새로 학습된 모델과 현재 Production 모델의 성능을 비교합니다.
    - Step 5: `register_and_promote`: 검증을 통과한 모델을 Production 단계로 승격시킵니다.
    """
    cfg = Config()

    @task
    def compute_cutoff(logical_date=None) -> str:
        """
        Airflow의 논리적 실행 날짜(logical_date)를 기반으로,
        데이터를 가져올 기준 시각(UTC)을 계산합니다.
        한국 시간(KST) 자정을 기준으로 UTC로 변환하여 반환합니다.
        """
        logger.info("===== Task [compute_cutoff] started =====")
        t = _t0()
        
        # KST 자정 시각 계산
        d_kst = logical_date.in_timezone(pendulum.timezone("Asia/Seoul")).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # UTC로 변환
        cutoff_utc = d_kst.in_timezone("UTC").to_datetime_string()
        
        logger.info(f"Logical Date: {logical_date}, KST Midnight: {d_kst}, UTC Cutoff: {cutoff_utc}")
        logger.info("===== Task [compute_cutoff] finished in %s =====", _elapsed(t))
        return cutoff_utc

    @task
    def load_data(cutoff_utc: str) -> str:
        """
        PostgreSQL DB에서 sales_train, features_train 테이블을 로드하여
        하나의 학습용 CSV 파일로 저장합니다.
        """
        logger.info("===== Task [load_data] started =====")
        import psycopg2
        t = _t0()

        # 저장할 디렉토리 및 파일 경로 설정
        data_dir = Path(cfg.data_dir) / "merged"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_cutoff = cutoff_utc.replace(":", "").replace(" ", "_")
        out_path = data_dir / f"sales_train_{safe_cutoff}.csv"
        logger.info(f"Output path: {out_path}")

        # DB에서 데이터 로드
        try:
            with psycopg2.connect(cfg.project_db_dsn) as conn:
                sales = pd.read_sql("SELECT * FROM sales_train", conn)
                features = pd.read_sql("SELECT * FROM features_train", conn)
            logger.info(f"Loaded sales: {len(sales)} rows, features: {len(features)} rows")
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            raise AirflowException("DB Connection or SQL query failed.")

        # 데이터 병합
        if "is_holiday" in features.columns:
            features = features.drop(columns=["is_holiday"])
        df = pd.merge(left=sales, right=features, on=["store", "date"])
        logger.info(f"Merged df shape: {df.shape}, Null count: {df.isna().sum().sum()}")

        # CSV 파일로 저장
        df.to_csv(out_path, index=False)
        logger.info("===== Task [load_data] finished in %s =====", _elapsed(t))
        return str(out_path)

    @task
    def train(uri: str) -> dict:
        """
        전달받은 데이터(uri)로 모델을 학습하고, 결과를 MLflow에 로깅합니다.
        내부적으로 데이터 전처리, 모델 학습, MLflow 로깅의 세 단계로 구성됩니다.
        """
        logger.info("===== Task [train] started =====")
        from lightgbm import LGBMRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        from sklearn.model_selection import train_test_split
        t = _t0()

        # --- 1. 데이터 전처리 ---
        def _preprocess_data(path: str):
            logger.info("--- Start: Data Preprocessing ---")
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            logger.info(f"Loaded df shape: {df.shape}, date range: [{df['date'].min()} → {df['date'].max()}]")
            
            # 시간 관련 피처 생성
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            # 휴일 피처를 0/1로 변환
            df["is_holiday"] = df["is_holiday"].apply(lambda x: 1 if x else 0)
            
            X = df.drop(columns=["weekly_sales", "date"])
            y = df["weekly_sales"]
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info(f"Split: train={len(y_train)}, val={len(y_test)}, features={X.shape[1]}")
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logger.info("Applied StandardScaler to features")
            logger.info("--- End: Data Preprocessing ---")
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

        # --- 2. 모델 학습 및 평가 ---
        def _train_model(X_train, y_train, X_test, y_test):
            logger.info("--- Start: Model Training & Evaluation ---")
            model = LGBMRegressor(**cfg.model_params)
            model.fit(X_train, y_train)
            logger.info("Model fit complete")

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            logger.info(f"Metrics: MAE={mae:.5f}, RMSE={rmse:.5f}")
            logger.info("--- End: Model Training & Evaluation ---")
            return model, {"val_mae": mae, "val_rmse": rmse}

        # --- 3. MLflow에 결과 로깅 ---
        def _log_to_mlflow(model, scaler, features, metrics, params, run_name):
            logger.info("--- Start: Logging to MLflow ---")
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment_name)

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                # 전처리기와 모델을 함께 패키징하여 로깅
                pyfunc_model = SklearnModelWrapper(model=model, scaler=scaler, features=features)
                mlflow.pyfunc.log_model(artifact_path="model", python_model=pyfunc_model)

                # 모델을 레지스트리에 등록
                mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/model",
                    name=cfg.mlflow_experiment_name,
                )
                logger.info(f"Logged and registered run={run_name} to MLflow.")
            logger.info("--- End: Logging to MLflow ---")
            return run.info.run_id

        # --- 메인 로직 실행 ---
        path = uri.replace("file://", "")
        X_train, X_test, y_train, y_test, scaler, feature_names = _preprocess_data(path)
        model, metrics = _train_model(X_train, y_train, X_test, y_test)

        run_name = f"{Path(uri).stem}_lgbm"
        all_metrics = {**metrics, "n_train": len(y_train), "n_val": len(y_test)}
        all_params = {
            **cfg.model_params,
            "algo": model.__class__.__name__,
            "scaler": "StandardScaler",
            "split": "random 80/20",
        }
        
        run_id = _log_to_mlflow(model, scaler, feature_names, all_metrics, all_params, run_name)

        logger.info("===== Task [train] finished in %s =====", _elapsed(t))
        return {
            "model_run_id": str(run_id),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_test)),
            "val_mae": float(metrics["val_mae"]),
            "val_rmse": float(metrics["val_rmse"]),
        }

    @task(do_xcom_push=False)
    def validate(train_meta: dict) -> None:
        """
        새로 학습된 모델과 현재 Production 모델의 성능을 비교하고,
        JSON-safe 결과만 수동으로 XCom에 푸시합니다.
        """
        import math 
        print("===== Task [validate] started =====")
        ctx = get_current_context()
        ti = ctx["ti"]

        # 1) 입력값을 파이썬 기본형으로 강제 + NaN/Inf 방지
        new_rmse = float(train_meta.get("val_rmse", 1e12))
        if not math.isfinite(new_rmse):
            new_rmse = 1e12

        experiment_name = cfg.mlflow_experiment_name
        client = MlflowClient()

        baseline_rmse = None
        try:
            # 현재 Production 단계 모델 가져오기
            prod_versions = client.get_latest_versions(experiment_name, stages=["Production"])
            if prod_versions:
                latest_prod_run = client.get_run(prod_versions[0].run_id)
                br = latest_prod_run.data.metrics.get("val_rmse")
                if br is not None:
                    baseline_rmse = float(br)
                    if not math.isfinite(baseline_rmse):
                        baseline_rmse = None
        except MlflowException as e:
            logger.warning(f"[validate] Could not fetch production model from MLflow: {e}. Assuming no baseline.")

        # 2) 개선 여부/비율 계산
        improved = (baseline_rmse is None) or (new_rmse < baseline_rmse)
        improve_ratio = 1.0 if baseline_rmse is None else (baseline_rmse - new_rmse) / baseline_rmse

        logger.info(
            f"[validate] {'Improved' if improved else 'Not Improved'}. "
            f"Baseline: {baseline_rmse}, New: {new_rmse}, Improvement: {improve_ratio:.2%}"
        )

        # 3) 수동 XCom 푸시 (JSON-safe 스칼라만)
        result = {
            "ok": bool(improved),
            "reason": f"{'improved' if improved else 'not_improved'}({improve_ratio:.2%})",
            "new_rmse": float(new_rmse),
            "baseline_rmse": None if baseline_rmse is None else float(baseline_rmse),
        }
        ti.xcom_push(key="validate_result", value=result)
        ti.xcom_push(key="validate_ok", value=bool(improved))  # 필요시 간단 키도 함께

        # 4) 자동 XCom 반환은 안 함
        return None

    @task
    def register_and_promote(train_meta: dict) -> dict:
        """
        검증을 통과한 모델을 Production 단계로 승격시킵니다.
        """
        ti = get_current_context()["ti"]
        validation = ti.xcom_pull(task_ids="validate", key="validate_result")
        print(validation)
        if not validation.get("ok", False):
            logger.warning("[promote] Skipping promotion as model did not improve.")
            return {
                "promoted": False,
                "reason": validation.get("reason", "validation_failed"),
            }

        model_name = cfg.mlflow_experiment_name
        new_rmse = validation["new_rmse"]
        client = MlflowClient()

        # 가장 최근에 등록된 모델 버전을 가져옴 (log_to_mlflow에서 방금 등록한 버전)
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

        # 기존 Production 모델들을 Archived로 변경
        for v in client.get_latest_versions(model_name, stages=["Production"]):
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )
            logger.info(f"[promote] Archived model version: {v.version}")

        # 새 모델을 Production으로 승격
        client.transition_model_version_stage(
            name=model_name, version=latest_version.version, stage="Production"
        )
        logger.info(
            f"[promote] Promoted model version {latest_version.version} to Production."
        )

        return {"promoted": True, "version": int(latest_version.version), "rmse": float(new_rmse)}

    cutoff = compute_cutoff()
    raw_uri = load_data(cutoff)
    train_meta = train(raw_uri)
    val_task = validate(train_meta)
    promote_task = register_and_promote(train_meta)
    
    val_task >> promote_task


daily_train_sales()
