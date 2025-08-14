import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
import logging
import mlflow
from typing import List

# --- 로거 설정 ---
logger = logging.getLogger("uvicorn.error")

# --- 환경 변수 및 설정 ---
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
# Airflow DAG의 `mlflow_experiment_name`과 일치시켜야 합니다.
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "sales_daily")

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title="Sales Prediction API",
    description="MLflow 모델 레지스트리에서 서빙되는 판매량 예측 모델 API",
    version="1.0.0",
)

# 모델을 저장할 전역 변수
model = None


# --- Pydantic 모델 정의 ---
class PredictionInput(BaseModel):
    """
    단일 예측 요청을 위한 입력 데이터 모델.
    Pydantic을 사용하여 타입 힌트 및 자동 유효성 검사를 수행합니다.
    """

    store: int = Field(..., example=1, description="상점 ID")
    dept: int = Field(..., example=1, description="부서 ID")
    date: str = Field(
        ..., example="2025-08-12", description="예측 기준 날짜 (YYYY-MM-DD)"
    )
    is_holiday: bool = Field(..., example=False, description="휴일 여부")
    temperature: float = Field(..., example=25.5, description="평균 기온")
    fuel_price: float = Field(..., example=3.8, description="연료 가격")
    cpi: float = Field(..., example=220.5, description="소비자 물가 지수")
    unemployment: float = Field(..., example=8.0, description="실업률")


class PredictionOutput(BaseModel):
    """단일 예측 결과 응답 모델"""

    prediction: float = Field(..., example=15000.50, description="예측된 주간 판매량")


class BatchPredictionOutput(BaseModel):
    """일괄 예측 결과 응답 모델"""

    predictions: List[float] = Field(
        ..., example=[15000.50, 22000.0], description="예측된 주간 판매량 리스트"
    )


# --- 모델 로딩 ---
@app.on_event("startup")
def load_production_model():
    """
    API 서버가 시작될 때 MLflow 모델 레지스트리에서 'Production' 단계의 모델을 로드합니다.
    모델 로딩에 실패하면 API는 시작되지 않고 에러를 발생시킵니다.
    """
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_EXPERIMENT_NAME}/Production"

    logger.info(f"Attempting to load model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model '{MLFLOW_EXPERIMENT_NAME}/Production' loaded successfully.")
    except Exception as e:
        logger.error(
            f"Fatal: Error loading model from MLflow Registry. API cannot start. Error: {e}"
        )
        # 모델 로딩 실패는 심각한 문제이므로, 예외를 다시 발생시켜 서버 시작을 중단시킬 수 있습니다.
        raise RuntimeError(f"Could not load model from MLflow: {e}") from e


# --- 전처리 함수 ---
def preprocess(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 데이터를 모델이 예측할 수 있는 형태로 전처리합니다.
    - `date` 컬럼에서 `year`, `month` 피처를 생성합니다.
    - `is_holiday`를 boolean에서 0/1 정수형으로 변환합니다.
    - 모델 학습 시 사용되지 않은 원본 `date` 컬럼을 제거합니다.

    참고: 스케일링 및 피처 순서 정렬은 MLflow에 저장된 `SklearnModelWrapper`가 자동으로 처리합니다.
    """
    # 1. date 컬럼 타입 변환 및 피처 생성
    df = input_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # 2. is_holiday 변환
    df["is_holiday"] = df["is_holiday"].astype(int)

    # 3. 모델 예측에 사용할 DataFrame 준비 (원본 date 컬럼 제외)
    final_features_df = df.drop(columns=["date"])

    return final_features_df


# --- API 엔드포인트 ---
@app.get("/", summary="헬스 체크")
def read_root():
    """API 서버의 상태와 모델 로드 여부를 확인하는 기본 엔드포인트입니다."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput, summary="단일 판매량 예측")
def predict(input_data: PredictionInput):
    """
    단일 데이터 포인트를 입력받아 주간 판매량을 예측합니다.
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded or failed to load."
        )

    try:
        # 입력 데이터를 DataFrame으로 변환
        input_df = pd.DataFrame([input_data.dict()])
        # 데이터 전처리
        processed_df = preprocess(input_df)
        # 모델 예측
        prediction = model.predict(processed_df)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        logger.exception(
            f"Prediction failed for input: {input_data.dict()}. Error: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during prediction: {e}",
        )


@app.post(
    "/predict_batch", response_model=BatchPredictionOutput, summary="일괄 판매량 예측"
)
def predict_batch(input_data: conlist(PredictionInput, min_length=1)):
    """
    여러 데이터 포인트를 리스트로 입력받아 각 항목의 주간 판매량을 일괄 예측합니다.
    Streamlit 대시보드 등에서 여러 샘플을 한 번에 테스트할 때 유용합니다.
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded or failed to load."
        )

    try:
        # 입력 데이터를 DataFrame으로 변환
        print(input_data)
        input_df = pd.DataFrame([d.dict() for d in input_data])
        # 데이터 전처리
        processed_df = preprocess(input_df)
        # 모델 예측
        predictions = model.predict(processed_df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        logger.exception(f"Batch prediction failed. Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during batch prediction: {e}",
        )
