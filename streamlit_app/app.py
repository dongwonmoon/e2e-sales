import streamlit as st
import pandas as pd
import psycopg2
import pendulum
import mlflow
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# --- 페이지 및 기본 설정 ---
st.set_page_config(
    page_title="Sales Forecasting MLOps Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 설정 중앙화 ---
# 앱 전체에서 사용될 설정들을 딕셔너리로 관리하여 일관성과 유지보수성을 높입니다.
APP_CONFIG = {
    "db_dsn": f"postgresql://{os.environ.get('PROJECT_DB_USER', 'sales')}:{os.environ.get('PROJECT_DB_PASSWORD', 'sales')}@{os.environ.get('PROJECT_DB_HOST', 'project-db')}:{os.environ.get('PROJECT_DB_PORT', 5432)}/{os.environ.get('PROJECT_DB_NAME', 'sales')}",
    "mlflow_tracking_uri": os.environ.get(
        "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
    ),
    "prediction_api_url": os.environ.get(
        "PREDICTION_API_URL", "http://prediction-service:8000"
    ),
    "mlflow_experiment_name": os.environ.get("MLFLOW_EXPERIMENT_NAME", "sales_daily"),
}

# MLflow 클라이언트 설정
mlflow.set_tracking_uri(APP_CONFIG["mlflow_tracking_uri"])


# --- 데이터 관련 함수 ---
@st.cache_data(ttl=30)
def query_db(sql: str) -> pd.DataFrame:
    """데이터베이스에 쿼리를 실행하고 결과를 DataFrame으로 반환합니다."""
    try:
        with psycopg2.connect(APP_CONFIG["db_dsn"]) as conn:
            return pd.read_sql(sql, conn)
    except psycopg2.OperationalError as e:
        st.error(f"데이터베이스 연결 실패: {e}", icon="🚨")
        return pd.DataFrame()


def inject_next_week_data(current_time: pendulum.DateTime) -> int:
    """스트림 데이터를 학습용 테이블에 주입하고, 주입된 행의 수를 반환합니다."""
    next_time = current_time.add(days=7)
    injected_rows = 0
    with psycopg2.connect(APP_CONFIG["db_dsn"]) as conn:
        with conn.cursor() as cursor:
            # 주입할 데이터가 있는지 먼저 확인
            cursor.execute(
                f"SELECT COUNT(*) FROM sales_stream WHERE date > '{current_time}' AND date <= '{next_time}'"
            )
            if cursor.fetchone()[0] > 0:
                # 데이터 주입
                cursor.execute(
                    f"INSERT INTO sales_train SELECT * FROM sales_stream WHERE date > '{current_time}' AND date <= '{next_time}'"
                )
                injected_rows = cursor.rowcount
                cursor.execute(
                    f"INSERT INTO features_train SELECT * FROM features_stream WHERE date > '{current_time}' AND date <= '{next_time}'"
                )
                cursor.execute(
                    f"UPDATE state SET cur_time = '{next_time}' WHERE id = 1"
                )
                conn.commit()
    return injected_rows


# --- UI 렌더링 함수 (탭별 분리) ---
def render_sidebar():
    """사이드바 UI를 렌더링합니다."""
    with st.sidebar:
        st.title("🌊 스트림 시뮬레이터")
        if "injected_data" not in st.session_state:
            st.session_state.injected_data = pd.DataFrame()

        cur_time_df = query_db("SELECT cur_time FROM state WHERE id = 1;")
        if not cur_time_df.empty:
            cur_time = pendulum.parse(str(cur_time_df["cur_time"][0]))
            st.info(f"현재 시뮬레이션 시각:\n**{cur_time.to_datetime_string()}**")

            if st.button("▶️ 다음 1주일 데이터 주입", use_container_width=True):
                # 데이터 주입 로직 실행
                injected_rows = inject_next_week_data(cur_time)
                if injected_rows > 0:
                    st.success(f"{injected_rows}개 판매 데이터 주입 완료!")
                    # 예측 테스트를 위해 주입된 데이터 조회
                    next_time = cur_time.add(days=7)
                    injected_sql = f"""
                    SELECT sa.weekly_sales, sa.dept, f.* FROM features_stream f 
                    JOIN sales_stream sa ON f.store = sa.store AND f.date = sa.date 
                    WHERE f.date > '{cur_time}' AND f.date <= '{next_time}';
                    """
                    st.session_state.injected_data = query_db(injected_sql)
                else:
                    st.warning("더 이상 주입할 데이터가 없습니다.")

                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("DB 상태 정보를 찾을 수 없습니다.")

        st.header("🔗 유용한 링크")
        st.page_link("http://localhost:8080", label="Airflow UI", icon="💨")
        st.page_link("http://localhost:5000", label="MLflow UI", icon="🔬")


def render_dashboard_tab():
    """'시스템 현황' 탭의 UI를 렌더링합니다."""
    st.header("종합 현황")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗃️ 데이터셋 크기")
        train_count_df = query_db("SELECT COUNT(*) FROM sales_train;")
        stream_count_df = query_db("SELECT COUNT(*) FROM sales_stream;")
        train_count = train_count_df["count"][0] if not train_count_df.empty else 0
        stream_count = stream_count_df["count"][0] if not stream_count_df.empty else 0

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=train_count,
                title={"text": "학습 데이터셋"},
                gauge={"axis": {"range": [None, train_count + stream_count]}},
            )
        )
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 프로덕션 모델")
        try:
            client = mlflow.tracking.MlflowClient()
            prod_model = client.get_latest_versions(
                APP_CONFIG["mlflow_experiment_name"], stages=["Production"]
            )
            if prod_model:
                latest_version = prod_model[0]
                run = client.get_run(latest_version.run_id)
                rmse = run.data.metrics.get("val_rmse", 0)
                fig = go.Figure(
                    go.Indicator(
                        mode="number+delta",
                        value=rmse,
                        title={"text": "RMSE"},
                        delta={
                            "reference": 14000,
                            "relative": False,
                            "valueformat": ".2f",
                        },
                    )
                )
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Version: {latest_version.version} / Run: `{latest_version.run_id[:7]}`"
                )
            else:
                st.warning("Production 모델 없음")
        except Exception as e:
            st.error(f"MLflow 연결 실패: {e}")

    st.divider()
    st.header("모델 성능 히스토리 (RMSE)")
    try:
        runs_df = mlflow.search_runs(
            experiment_names=[APP_CONFIG["mlflow_experiment_name"]],
            order_by=["start_time DESC"],
        )
        if not runs_df.empty:
            fig = px.line(
                runs_df,
                x="start_time",
                y="metrics.val_rmse",
                title="RMSE 변화 추이",
                markers=True,
                labels={"start_time": "학습 시간", "metrics.val_rmse": "RMSE"},
            )
            fig.update_layout(yaxis_title="RMSE", xaxis_title="시간")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("기록된 학습 Run이 없습니다.")
    except Exception as e:
        st.error(f"MLflow Run 기록 조회 실패: {e}")


def render_prediction_test_tab():
    """'모델 예측 테스트' 탭의 UI를 렌더링합니다."""
    st.header("방금 주입한 데이터로 예측 테스트")
    if not st.session_state.injected_data.empty:
        st.info(
            "사이드바에서 주입한 최신 데이터 중 5개를 샘플링하여 예측을 수행합니다."
        )

        sample_df = st.session_state.injected_data.sample(
            min(5, len(st.session_state.injected_data))
        )

        # API에 전송할 페이로드 생성
        payload = sample_df.rename(columns={"isholiday": "is_holiday"}).to_dict(
            orient="records"
        )
        # 날짜 형식 변환
        for item in payload:
            item["date"] = item["date"].strftime("%Y-%m-%d")
        try:
            # 일괄 예측 API 호출
            api_url = f"{APP_CONFIG['prediction_api_url']}/predict_batch"
            response = requests.post(api_url, json=payload)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

            predictions = response.json()["predictions"]
            sample_df["predicted_sales"] = predictions

            # 결과 비교 테이블 및 차트 표시
            display_cols = ["date", "store", "dept", "weekly_sales", "predicted_sales"]
            st.dataframe(
                sample_df[display_cols].rename(
                    columns={
                        "weekly_sales": "실제 판매량",
                        "predicted_sales": "예측 판매량",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        except requests.exceptions.RequestException as e:
            st.error(f"API 요청 실패: {e}", icon="🔥")
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")

    else:
        st.warning("먼저 사이드바에서 데이터를 주입해주세요.")


def render_logs_tab():
    """'학습 로그' 탭의 UI를 렌더링합니다."""
    st.header("MLflow 최신 학습 로그")
    st.info("가장 최근에 실행된 학습(Run)의 상세 정보입니다.")
    try:
        runs_df = mlflow.search_runs(
            experiment_names=[APP_CONFIG["mlflow_experiment_name"]],
            max_results=1,
            order_by=["start_time DESC"],
        )
        if not runs_df.empty:
            latest_run_id = runs_df.iloc[0]["run_id"]
            run = mlflow.get_run(latest_run_id)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Metrics")
                st.json(run.data.metrics)
            with c2:
                st.subheader("Parameters")
                st.json(run.data.params)
        else:
            st.info("기록된 학습 Run이 없습니다.")
    except Exception as e:
        st.error(f"MLflow Run 기록 조회 실패: {e}")


# --- 메인 애플리케이션 실행 ---
def main():
    """메인 함수: 전체 애플리케이션을 구성하고 실행합니다."""
    st.title("📊 MLOps 모니터링 대시보드")

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(
        ["**📈 시스템 현황**", "**🤖 모델 예측 테스트**", "**📜 학습 로그**"]
    )

    with tab1:
        render_dashboard_tab()
    with tab2:
        render_prediction_test_tab()
    with tab3:
        render_logs_tab()


if __name__ == "__main__":
    main()
