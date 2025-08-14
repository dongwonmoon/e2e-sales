import mlflow

# ========== MLflow 모델 래퍼 ==========
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Scikit-learn 모델과 전처리 단계를 함께 감싸는 MLflow 래퍼 클래스.
    - model: 학습된 머신러닝 모델 (예: LGBMRegressor)
    - scaler: 피처 스케일링에 사용된 객체 (예: StandardScaler)
    - features: 모델 학습에 사용된 피처 이름 리스트
    이 클래스를 통해 모델을 저장하면, 예측 시 동일한 피처 순서와 스케일링을 보장할 수 있습니다.
    """
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features

    def predict(self, context, model_input):
        """
        예측 요청이 들어왔을 때 실행되는 메소드.
        1. 입력 데이터(model_input)의 컬럼을 학습 시 사용된 피처 순서대로 정렬합니다.
        2. 저장된 스케일러(self.scaler)를 사용해 데이터를 변환합니다.
        3. 변환된 데이터로 예측을 수행하고 결과를 반환합니다.
        """
        # 입력 DataFrame에서 학습에 사용된 피처 순서대로 정렬
        model_input = model_input[self.features]
        # 스케일러 적용
        scaled_input = self.scaler.transform(model_input)
        # 예측
        return self.model.predict(scaled_input)
