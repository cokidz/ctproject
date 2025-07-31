import streamlit as st
import streamlit.components.v1 as htmlviewer
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(layout='wide', page_title='This is WebApp.')

# 타이틀
st.title('This is WebApp.')


# 메인 컬럼 (4:1 비율 유지)
col1, col2 = st.columns((4,1))

with col1:
    # 1번째 익스팬더와 Tips를 상단에 고정하기 위해 서브 컬럼 생성
    subcol1, subcol2 = st.columns((4,1))  # 메인 비율과 동일하게 설정
    
    with subcol1:
        with st.expander('Ex-Content #1'):
            url = "https://youtu.be/UTUsmzI0jBA?si=p4LlOE6t1ZsPryED"
            st.info('info-Content..')
            st.video(url)
    
    with subcol2:
        with st.expander('Tips..'):
            st.info('Tips..')
    
    
    
    # 나머지 익스팬더들은 아래에 순서대로 배치
    with st.expander('비버 챌린지 #1'):
        with open('./pb1.html','r', encoding='utf-8') as f:
            html_pb1 = f.read()
            f.close()
        htmlviewer.html(html_pb1, height=700,scrolling=True)

    with st.expander('비버 챌린지 #2'):
        with open('./pb2.html','r', encoding='utf-8') as f:
            html_pb2 = f.read()
            f.close()
        htmlviewer.html(html_pb2, height=700,scrolling=True)

    with st.expander('AI applied CT_PB'):
        try:
            # ct_pb.py 파일 읽기
            with open('./ct_pb.py', 'r', encoding='utf-8') as f:
                py_code = f.read()
            
            # exec로 코드 실행
            exec(py_code, globals())
            
            st.write("")
        except Exception as e:
            st.error(f": {str(e)}")

# col2는 이제 비어 있음 (필요 없으면 삭제 가능)

st.markdown('<hr>', unsafe_allow_html=True)
st.write('<font color="BLUE">(C)Copyright. all rights reserved by wondolMT', unsafe_allow_html=True)
