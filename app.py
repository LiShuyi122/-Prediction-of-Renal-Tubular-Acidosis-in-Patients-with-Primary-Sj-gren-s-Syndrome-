import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="原发性干燥综合征患者肾小管性酸中毒预测",
    page_icon=":hospital:",
    layout="centered"
)

# 应用标题
st.title("原发性干燥综合征患者肾小管性酸中毒预测")
st.markdown("输入患者指标数据，预测肾小管性酸中毒风险")

# 缓存模型加载，提高性能
@st.cache_resource
def load_model():
    try:
        # 加载模型和标准化器
        model = joblib.load('gbdt_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None

# 加载模型
model, scaler = load_model()

# 创建输入表单
with st.form("prediction_form"):
    st.subheader("患者指标输入")
    
    # 定义表单输入字段
    col1, col2 = st.columns(2)
    
    with col1:
        alt = st.number_input("ALT值 (U/L)", min_value=0.0, step=0.1)
        albumin = st.number_input("白蛋白值 (g/L)", min_value=0.0, step=0.1)
        hemoglobin = st.number_input("血红蛋白值 (g/L)", min_value=0.0, step=0.1)
        
    with col2:
        erythrocyte_sedimentation = st.number_input("血沉值 (mm/h)", min_value=0.1, step=0.1)
        antibody = st.selectbox("抗合成酶抗体阳性", ["否 (0)", "是 (1)"])
        triglyceride = st.number_input("甘油三酯值 (mmol/L)", min_value=0.0, step=0.1)
    
    # 转换抗体选择为数值
    antibody_value = 1 if antibody == "是 (1)" else 0
    
    # 提交按钮
    submitted = st.form_submit_button("提交预测")
    
    if submitted:
        if model and scaler:
            try:
                # 验证输入
                if erythrocyte_sedimentation == 0:
                    st.error("血沉值不能为0")
                else:
                    # 构建输入数据
                    input_data = {
                        'ALT': alt,
                        '血沉': erythrocyte_sedimentation,
                        'ALT_÷_血沉': alt / erythrocyte_sedimentation,
                        '白蛋白': albumin,
                        '抗合成酶抗体阳性': antibody_value,
                        '血红蛋白': hemoglobin,
                        '甘油三酯': triglyceride
                    }
                    
                    # 转换为DataFrame
                    X_new = pd.DataFrame([input_data])[['ALT_÷_血沉', '白蛋白', '抗合成酶抗体阳性', '血红蛋白', '甘油三酯']]
                    
                    # 标准化
                    X_new_scaled = scaler.transform(X_new)
                    
                    # 预测概率
                    prob = model.predict_proba(X_new_scaled)[0][1]
                    
                    # 确定分级
                    grade = 1 if prob >= 0.5 else 0
                    grade_text = f"ILD分级为{grade}级"
                    
                    # 显示结果
                    st.subheader("预测结果")
                    st.success(f"{grade_text}，预测概率：{prob * 100:.2f}%")
                    
                    # 可视化预测概率
                    st.progress(prob)
                    st.caption(f"预测概率: {prob * 100:.2f}%")
                    
                    # 提供解释
                    if grade == 1:
                        st.warning("预测结果显示患者有较高风险患有肾小管性酸中毒，建议进一步检查。")
                    else:
                        st.info("预测结果显示患者肾小管性酸中毒风险较低，但仍需结合临床症状综合判断。")
                    
            except Exception as e:
                st.error(f"预测过程中出错: {e}")
        else:
            st.error("模型未正确加载，无法进行预测。")

# 页脚信息
st.markdown("---")
st.caption("注意: 本预测结果仅供参考，不能替代专业医疗建议。")
st.caption("© 2024 原发性干燥综合征患者肾小管性酸中毒预测系统")
