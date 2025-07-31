import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# Wide mode 설정 추가
st.set_page_config(layout="wide")  # 전체 앱을 Wide mode로 설정하여 더 넓은 레이아웃 제공

# KMA API 설정
KMA_AUTH_KEY = st.secrets["kma"]["KMA_AUTH_KEY"]  # secrets.toml에서 KMA_AUTH_KEY 가져오기
DOMAIN = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"
STN = "113"  # 공주 지역 stn ID

# 샘플 데이터 (초기 표시용, 7일)
SAMPLE_TEMPS = [24.5, 27.2, 25.9, 25.1, 26.3, 27.5, 28.8]
SAMPLE_DATES = ["2024-07-17", "2024-07-18", "2024-07-19", "2024-07-20", 
                "2024-07-21", "2024-07-22", "2024-07-23"]

# 온도 데이터 가져오기 함수
def fetch_temperatures(start_date, days):
    temperatures = []
    dates = []
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        tm = current_date.strftime("%Y%m%d")
        url = f"{DOMAIN}?tm={tm}&stn={STN}&disp=0&help=0&authKey={KMA_AUTH_KEY}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                lines = response.text.splitlines()
                for line in lines:
                    if not line.startswith('#') and line.strip():
                        data = line.split()
                        ta_max_index = 10  # TA_MAX 인덱스 (API 문서 확인 필요)
                        ta_max = float(data[ta_max_index])
                        temperatures.append(ta_max)
                        dates.append(current_date.strftime("%Y-%m-%d"))
                        break
                else:
                    st.warning(f"{tm} 데이터가 없습니다.")
                    temperatures.append(None)
                    dates.append(current_date.strftime("%Y-%m-%d"))
            else:
                st.error(f"{tm} API 호출 오류: {response.status_code}")
        except Exception as e:
            st.error(f"{tm} 데이터 가져오기 오류: {e}")
    
    return dates, temperatures

# 다음 날 실제 기온 가져오기 함수
def fetch_next_day_temperature(start_date, days):
    next_date = start_date + timedelta(days=days)
    tm = next_date.strftime("%Y%m%d")
    url = f"{DOMAIN}?tm={tm}&stn={STN}&disp=0&help=0&authKey={KMA_AUTH_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            for line in lines:
                if not line.startswith('#') and line.strip():
                    data = line.split()
                    ta_max_index = 10  # TA_MAX 인덱스
                    return float(data[ta_max_index])
            st.warning(f"{tm} 데이터가 없습니다.")
            return None
        else:
            st.error(f"{tm} API 호출 오류: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"{tm} 데이터 가져오기 오류: {e}")
        return None

# Streamlit 앱
st.title("공주시는 몇 도씨?")
st.write("공주 지역의 과거 최고 기온 데이터를 보고 다음 날 최고 기온을 예측해 봅시다!")

# 초기 데이터 설정 (샘플 데이터)
if 'temperatures' not in st.session_state:
    st.session_state['dates'] = SAMPLE_DATES
    st.session_state['temperatures'] = SAMPLE_TEMPS
    st.session_state['days'] = 7

# 시작일 및 데이터 기간 설정
start_date_str = st.date_input("시작일 선택 (YYYY-MM-DD)", value=datetime(2024, 7, 17))
days = st.slider("데이터 가져오기 기간 (일)", min_value=7, max_value=30, value=7)
start_date = datetime.combine(start_date_str, datetime.min.time())

# 데이터 불러오기 버튼
if st.button("데이터 불러오기"):
    dates, temperatures = fetch_temperatures(start_date, days)
    if temperatures:
        st.session_state['dates'] = dates
        st.session_state['temperatures'] = temperatures
        st.session_state['days'] = days
        st.write(f"데이터 날짜 범위: {dates[0]} ~ {dates[-1]}")
        
        # 데이터 업데이트 후 fig 재생성
        df = pd.DataFrame({'일수': range(1, days + 1), '날짜': dates, '기온 (°C)': temperatures})
        st.session_state['fig'] = go.Figure()
        st.session_state['fig'].add_trace(go.Scatter(
            x=df['일수'], y=df['기온 (°C)'], mode='markers+lines', name='기온 데이터',
            marker=dict(size=10), text=df['날짜'], hovertemplate='%{text}<br>기온: %{y}°C'
        ))
        st.session_state['fig'].update_layout(
            title='기온 데이터',
            xaxis_title='일수',
            yaxis_title='최고 기온 (°C)',
            xaxis_range=[0.5, days + 1.5],
            yaxis=dict(autorange=True),
            dragmode='lasso',
            margin=dict(l=50, r=50, t=50, b=50)
        )
    else:
        st.error("데이터를 불러오지 못했습니다. 샘플 데이터를 계속 사용합니다.")

# 데이터 표시
temps = st.session_state['temperatures']
dates = st.session_state['dates']
days = st.session_state['days']
df = pd.DataFrame({'일수': range(1, days + 1), '날짜': dates, '기온 (°C)': temps})

# 데이터 표 표시
st.subheader("가져온 기온 데이터")
st.dataframe(df[['날짜', '기온 (°C)']], use_container_width=True)

# 평균, 중앙값, 표준편차 표시
valid_temps = [t for t in temps if t is not None]
if valid_temps:
    mean_temp = np.mean(valid_temps)
    median_temp = np.median(valid_temps)
    std_temp = np.std(valid_temps, ddof=1)  # 표본 표준편차
    st.write(f"**평균 기온**: {mean_temp:.1f} °C")
    st.write(f"**중앙값 기온**: {median_temp:.1f} °C")
    st.write(f"**표준편차**: {std_temp:.1f} °C")
else:
    st.write("**평균 기온**: 데이터 없음")
    st.write("**중앙값 기온**: 데이터 없음")
    st.write("**표준편차**: 데이터 없음")

# 그래프 초기화 (세션 상태에 fig가 없으면 초기화)
if 'fig' not in st.session_state:
    st.session_state['fig'] = go.Figure()
    st.session_state['fig'].add_trace(go.Scatter(
        x=df['일수'], y=df['기온 (°C)'], mode='markers+lines', name='기온 데이터',
        marker=dict(size=10), text=df['날짜'], hovertemplate='%{text}<br>기온: %{y}°C'
    ))
    st.session_state['fig'].update_layout(
        title='기온 데이터',
        xaxis_title='일수',
        yaxis_title='최고 기온 (°C)',
        xaxis_range=[0.5, days + 1.5],
        yaxis=dict(autorange=True),
        dragmode='lasso',
        margin=dict(l=50, r=50, t=50, b=50)
    )

# 그래프 표시
st.plotly_chart(st.session_state['fig'], use_container_width=True)

# AI 예측 버튼
if st.button("AI 예측"):
    valid_data = [(i + 1, t) for i, t in enumerate(temps) if t is not None]
    if len(valid_data) >= 2:  # 선형 회귀를 위해 최소 2개 데이터 필요
        X = np.array([x[0] for x in valid_data]).reshape(-1, 1)
        y = np.array([x[1] for x in valid_data])
        model = LinearRegression()
        model.fit(X, y)
        next_day = np.array([[days + 1]])
        predicted_temp = model.predict(next_day)[0]
        st.session_state['prediction'] = predicted_temp
        next_date = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
        st.write(f"AI 예측: {next_date} 기온은 약 {predicted_temp:.1f} °C입니다. (선형 회귀 기반)")
        
        # 기존 추세선 제거
        new_data = [trace for trace in st.session_state['fig'].data 
                    if trace.name not in ['추세선', '추세선(선형회귀선)']]
        st.session_state['fig'] = go.Figure(
            data=new_data + [go.Scatter(
                x=[1, days + 1], 
                y=model.predict(np.array([1, days + 1]).reshape(-1, 1)), 
                mode='lines', 
                name='추세선(선형회귀선)',
                line=dict(color='red', dash='dash')
            )],
            layout=st.session_state['fig'].layout
        )
        st.rerun()  # 그래프 갱신
    else:
        st.error("선형 회귀 예측을 위해 최소 2개의 유효한 데이터가 필요합니다.")

# Update the layout
st.session_state['fig'].update_layout(
    title='기온 데이터',
    xaxis_title='일수',
    yaxis_title='최고 기온 (°C)',
    xaxis_range=[0.5, days + 1.5],
    yaxis=dict(autorange=True),
    dragmode='lasso',
    margin=dict(l=50, r=50, t=50, b=50)
)

# CT 기반 활동지
st.subheader("CT 기반 활동지")
st.write("CT를 활용해 최고 기온 데이터를 분석하고 다음 날 최고 기온을 예측해 보세요. 각 섹션에서 왼쪽은 활동, 오른쪽은 관련 CT 요소를 확인할 수 있습니다.")

# 섹션 1: 데이터 관찰과 추상화
col1, col2 = st.columns([4, 1])  # 컬럼 비율을 [4, 1]로 조정하여 왼쪽 컬럼을 더 넓게
with col1:
    with st.expander("섹션 1: 데이터 관찰과 추상화"):
        st.write("표와 그래프를 보고 기온 데이터의 패턴을 찾아보세요.")
        pattern_observation = st.text_area(
            "1.1 그래프와 표를 보고 기온 데이터에서 어떤 패턴이나 추세가 관찰되나요? (예: 기온이 점차 상승하거나, 특정 요일에 높음) 구체적인 패턴과 그 이유를 설명하세요.",
            placeholder="예: 기온이 점차 상승하는 추세를 보인다. 이유는 여름철로 접어들며 날씨가 따뜻해지고 있다."
        )
        features = st.multiselect(
            "1.2 다음 중 데이터에서 가장 두드러지는 특징은 무엇인가요? (복수 선택 가능)",
            ["기온이 점차 상승하는 추세", "특정 요일에 기온이 높음", "변동성이 큼(표준편차 기반)", "이상치(극단값)가 있음","기타"],
            help="그래프에서 추세선이나 데이터 분포를 보고 선택하세요."
        )
        features_reason = st.text_area(
            "1.3 선택한 특징의 근거를 데이터를 바탕으로 설명하세요.",
            placeholder="예: 날짜별로 기온을 그래프로 시각화하여 만약 상승 추세가 있는 경우 날짜가 증가할수록 기온이 높아지는 점들로 확인될 수 있다."
        )
with col2:
    with st.expander("CT 요소"):
        st.markdown("""
        - **데이터로서 코드 이해하기**: 표와 그래프는 `pandas`와 `plotly` 코드를 통해 데이터를 시각화. 날짜와 기온을 구조화된 데이터로 저장.
        - **추상화하기**: 복잡한 기온 데이터를 상승/하강 추세 같은 간단한 패턴으로 요약.
        """)

# 섹션 2: 데이터 분해와 가설 수립
col1, col2 = st.columns([4, 1])  # 컬럼 비율 조정
with col1:
    with st.expander("섹션 2: 데이터 분해와 가설 수립"):
        st.write(f"통계 데이터를 분석해 다음 날 최고 기온을 예측하기 위한 가설을 세워보세요. 각 평균({mean_temp:.1f} °C), 중앙값({median_temp:.1f} °C), 표준편차({std_temp:.1f} °C)를 참고하세요.")
        distribution_eval = st.text_area(
            f"2.1 평균({mean_temp:.1f} °C), 중앙값({median_temp:.1f} °C), 표준편차({std_temp:.1f} °C)를 바탕으로 데이터 분포를 평가하세요. (예: 데이터가 고르게 분포했나, 한쪽으로 치우쳤나?)",
            placeholder="예: 평균과 중앙값이 비슷하면 데이터가 고르게 분포했을 가능성이 크다. 표준편차가 크면 예측이 어려울 수 있다."
        )
        hypothesis = st.text_area(
            "2.2 주어진 데이터를 바탕으로 기온 변화에 대한 가설을 세우세요.",
            placeholder="예: 가설: 기온이 매일 0.5°C씩 상승한다. 근거: 그래프에서 선형 추세를 보임. 예측: 선형 회귀를 사용해 다음 날 예측."
        )
with col2:
    with st.expander("CT 요소"):
        st.markdown("""
        - **분해하기**: 복잡한 기온 예측 문제를 데이터 분석, 가설 설정, 예측으로 나누어 해결.
        - **모의로 생각하기**: 데이터 기반 가설을 세우고, 다음 날 기온을 예측하는 모의 실험 수행.
        """)

# 섹션 3: 예측과 단순화
next_date = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
col1, col2 = st.columns([4, 1])  # 컬럼 비율 조정
with col1:
    with st.expander(f"섹션 3: {next_date} 기온 예측과 단순화"):
        st.write(f"{next_date}의 기온을 예측하고, 복잡한 데이터를 간단한 방식으로 해석해 보세요.")
        user_prediction = st.number_input(
            f"3.1 {next_date} 기온을 예측하세요 (°C)", min_value=-50.0, max_value=50.0, value=25.0,
            help="그래프의 추세나 통계를 참고해 예측하세요."
        )
        user_reason = st.text_area(
            "3.2 예측 근거를 자세히 설명하세요. 최소 1가지 이상의 근거를 나열하고, 데이터를 간단히 해석한 방법을 포함하세요 (예: 평균 기온 사용, 최근 3일 추세).",
            placeholder="예: 1) 최근 3일 기온이 상승 추세. 2) 평균 기온이 26°C로 안정적. 3) 표준편차를 고려해 약간의 변동 예상."
        )
        uncertainty_range = st.text_input(
            "3.3 불확실성을 고려한 범위 예측을 입력하세요 (예: 25.0 ± 3.0 °C). 이 범위를 선택한 이유는 무엇인가요?",
            placeholder=f"예: {user_prediction:.1f} ± {std_temp:.1f} °C (표준편차 기반)",
            help="표준편차나 데이터 변동성을 참고해 범위를 설정하세요."
        )
with col2:
    with st.expander("CT 요소"):
        st.markdown("""
        - **단순하게 생각하기**: 복잡한 통계 데이터를 간단한 예측(예: 평균 기반)으로 표현.
        - **모의로 생각하기**: 데이터 기반 가설을 세우고, 다음 날 기온을 예측하는 모의 실험 수행.
        """)

# 예측 비교 버튼
if st.button("예측과 실제 기온 비교"):
    actual_temp = fetch_next_day_temperature(start_date, days)
    if actual_temp is not None:
        ai_pred = st.session_state.get('prediction', None)
        st.write(f"**실제 기온** ({next_date}): {actual_temp:.1f} °C")
        
        # 사용자 예측 비교
        user_diff = abs(actual_temp - user_prediction)
        st.write(f"**사용자 예측** ({user_prediction:.1f} °C) 차이: {user_diff:.1f} °C")
        
        # AI 예측 비교
        if ai_pred is not None:
            ai_diff = abs(actual_temp - ai_pred)
            st.write(f"**AI 예측** ({ai_pred:.1f} °C) 차이: {ai_diff:.1f} °C")
        else:
            st.warning("AI 예측을 먼저 실행해주세요.")
    else:
        st.error("실제 기온 데이터를 가져오지 못했습니다.")

# 섹션 4: 평가와 변형
col1, col2 = st.columns([4, 1])  # 컬럼 비율 조정
with col1:
    with st.expander("섹션 4: 평가와 변형"):
        st.write("예측 결과와 실제 데이터를 비교하고, AI 예측 코드의 설계를 분석해 보세요.")
        comparison_analysis = st.text_area(
            "4.1 실제 기온과 사용자 예측의 차이를 분석하세요. 차이가 발생한 원인은 무엇일까? (예: 데이터 부족, 패턴 오해, 변인 부족 등)",
            placeholder="예: 오차는 최근 데이터의 변동성을 고려하지 않아 발생. 더 긴 기간의 데이터를 분석하면 정확도가 높아질 수 있을 것으로 생각함."
        )
        ai_comparison = st.text_area(
            "4.2 AI 예측(선형 회귀)과 비교해 어느 쪽이 더 정확했나요? 선형 회귀 외에 다른 예측 방법(예: 이동 평균)을 제안하세요.",
            placeholder="예: AI는 선형 회귀를 사용했으며, 이동 평균을 사용하면 단기 변동을 줄일 수 있다."
        )
with col2:
    with st.expander("CT 요소"):
        st.markdown("""
        - **변형해서 생각하기**: 선형 회귀 외 다른 예측 방법(예: 이동 평균)을 제안해 변형된 접근법 탐구.
        - **중복 방지 생각하기**: AI 코드에서 추세선 중복을 방지하기 위해 기존 트레이스를 필터링하는 설계 이해.
        """)

# 섹션 5: 반성과 개선 (선택 사항)
col1, col2 = st.columns([4, 1])  # 컬럼 비율 조정
with col1:
    with st.expander("섹션 5: 반성과 개선 (선택 사항)"):
        st.write("예측 과정을 돌아보고, 더 간단하거나 효과적인 방법으로 개선할 방법을 제안하세요.")
        reflection = st.text_area(
            "5.1 이번 예측 과정을 통해 배운 점은 무엇인가요? 각 단계를 나누어 (데이터 수집, 분석, 예측) 어떤 점을 개선할 수 있을까요?",
            placeholder="예: 데이터 수집 단계에서 더 긴 기간을 분석하면 좋았을 것이다. 예측 단계에서는 더 간단한 모델을 시도할 수 있다."
        )
        extension = st.text_area(
            "5.2 만약 더 많은 데이터(예: 60일)를 사용하거나, 다른 변수를 추가(예: 습도)한다면 예측이 어떻게 달라질까요? 이유를 설명하세요.",
            placeholder="예: 60일 데이터는 장기 추세를 더 잘 보여줄 수 있다. 습도를 추가하면 기온 변화의 원인을 더 정확히 파악할 수 있다."
        )
with col2:
    with st.expander("CT 요소"):
        st.markdown("""
        - **분해하기**: 예측 과정을 데이터 수집, 분석, 예측 단계로 나누어 검토.
        - **단순하게 생각하기**: 복잡한 예측 문제를 간단한 개선 방안으로 해결.
        """)
