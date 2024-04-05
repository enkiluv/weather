# -*- coding: utf-8 -*-

# 패키지 가져오기
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# 로더 함수
@st.cache_resource
def load_context(dataname):
    state = wonder.load_state(f"{dataname}_state.pkl")
    model = wonder.input_piped_model(state)
    return state, model

# 드라이버 함수
if __name__ == "__main__":
    # 스트림릿 인터페이스
    st.subheader(f"날씨예측기")
    st.markdown(":blue[**AI Wonder**] 제공")

    # 사용자 입력
    개구리울음 = st.number_input("개구리울음", value=67)
    무릎통증 = st.number_input("무릎통증", value=76)

    st.markdown("")

    # 입력값으로 데이터 만들기
    point = pd.DataFrame([{
        '개구리울음': 개구리울음,
        '무릎통증': 무릎통증,
    }])

    # 컨텍스트 로드
    state, model = load_context('날씨예측')

    # 예측 및 설명
    if st.button('예측'):
        st.markdown("")

        with st.spinner("추론 중..."):
            prediction = str(model.predict(point)[0])
            st.success(f"**{state.target}**의 예측값은 **{prediction}** 입니다.")
            predprobas = zip(['맑음', '비', '흐림'],
                np.round(model.predict_proba(point)[0], 2))
            predprobas_str = ", ".join([f"{label}: {proba}" for label, proba in predprobas])
            st.success(f"각 클래스의 예측확률은 **{predprobas_str}** 입니다.")
            st.markdown("")

        with st.spinner("설명 생성 중..."):
            st.info("피처 중요도")
            importances = pd.DataFrame(wonder.local_explanations(state, point), columns=["피처", "값", "중요도"])
            st.dataframe(importances.round(2))

            st.info("반사실 추천")
            counterfactuals = wonder.whatif_instances(state, point).iloc[:20]
            st.dataframe(counterfactuals.round(2))
