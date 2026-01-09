import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="Maize Yield Predict (Lite)", 
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Maize Yield Predict")
st.markdown("""
**Model Version**: XGBoost.\n
Please input these factors: **Rainfall**, **Temperature**, and **Soil pH**.
""")

# ---------------------------------------------------------
# 2. 加载模型
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'xgb_model.pkl' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ---------------------------------------------------------
# 3. 用户输入 (仅展示有效的3个特征)
# ---------------------------------------------------------
if model:
    st.divider()
    st.subheader("📝 Key Environmental Inputs")
    
    input_data = {}
    
    # 使用三列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data['rainfall'] = st.slider(
            "🌧️ Rainfall (mm)", 
            min_value=0.0, max_value=500.0, value=140.0, step=1.0,
            help="Total rainfall during growing season"
        )
        
    with col2:
        input_data['temperature'] = st.slider(
            "🌡️ Temperature (°C)", 
            min_value=0.0, max_value=50.0, value=27.0, step=0.1,
            help="Average temperature"
        )
        
    with col3:
        input_data['soil_ph'] = st.slider(
            "🧪 Soil pH", 
            min_value=3.0, max_value=10.0, value=6.5, step=0.1,
            help="Soil acidity (3=Acidic, 10=Alkaline)"
        )

    st.divider()

    # ---------------------------------------------------------
    # 4. 预测逻辑 (包含自动特征工程)
    # ---------------------------------------------------------
    if st.button("🚀 Run Prediction", type="primary"):
        try:
            # 1. 创建基础 DataFrame
            df_input = pd.DataFrame([input_data])
            
            # 2. 自动生成模型需要的“平方项”
            # 模型实际上需要6列：[x, y, z, x^2, y^2, z^2]
            df_input['rainfall_sq'] = df_input['rainfall'] ** 2
            df_input['temperature_sq'] = df_input['temperature'] ** 2
            df_input['soil_ph_sq'] = df_input['soil_ph'] ** 2
            
            # 3. 严格按照模型报错信息中的顺序排列列
            expected_cols = [
                'rainfall', 'temperature', 'soil_ph', 
                'rainfall_sq', 'temperature_sq', 'soil_ph_sq'
            ]
            df_final = df_input[expected_cols]
            
            # 4. 进行预测
            prediction = model.predict(df_final)
            
            # 5. 展示结果
            st.success("✅ Prediction Complete")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(
                    label="Estimated Yield", 
                    value=f"{prediction[0]:.0f} kg/ha", 
                    # delta="kg/ha"
                )
            
            # with col_res2:
            #     # 简单的解释性文字
            #     val = prediction[0]

                # if val > 4800:
                #     st.success("🌟 Excellent Yield Potential")
                #     st.write("Conditions are near optimal for maize.")
                # elif val > 3800:
                #     st.info("✅ Good/Average Yield")
                #     st.write("Standard productivity expected.")
                # else:
                #     st.warning("⚠️ Low Yield Risk")
                #     st.write("Environmental stress factors detected.")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")