import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import Preprocessor

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Health Check",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #FAFAFA; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #2E86C1; }
    .stButton>button { background-color: #28B463; color: white; border-radius: 8px; height: 50px; font-size: 18px; border: none; margin-top:10px; }
    .stButton>button:hover { background-color: #239B56; }
    .big-font { font-size: 20px !important; font-weight: 500; color: #555; }
    .result-card { padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/final_model.pkl")
        prep = joblib.load("models/preprocessor.pkl")
        return model, prep
    except Exception as e:
        st.error(f"⚠️ Could not load the model. Error: {e}")
        return None, None

model, preprocessor = load_model()

# --- Centered Title ---
st.markdown("<h1 style='text-align:center;'>❤️ Diabetes Health Check</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;' class='big-font'>Welcome! Assess your potential diabetes risk based on health indicators.</p><hr>", unsafe_allow_html=True)

if not model:
    st.warning("Model files missing. Please run the training script first.")
    st.stop()

# --- Input Form ---
with st.form("health_check_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Personal Info & Lifestyle")
        age = st.number_input("Age (years)", 1, 120, 40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
        height = st.number_input("Height (cm)", 50.0, 250.0, 170.0)
        bmi = weight / ((height/100) ** 2)
        st.info(f"💡 Estimated BMI: {bmi:.1f}")
        alcohol = st.number_input("Alcohol consumption per week (units)", 0, 50, 0)
        sleep_hours = st.slider("Average sleep hours per day", 0, 24, 7)
        screen_time = st.slider("Screen time hours per day", 0, 24, 5)
        physical_activity = st.number_input("Physical activity minutes per week", 0, 1000, 150)
        diet_score = st.slider("Diet score (1-10)", 1, 10, 5)
        
    with col2:
        st.subheader("🩺 Health & Medical History")
        waist_to_hip = st.number_input("Waist-to-hip ratio", 0.4, 2.0, 0.85)
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 50, 150, 80)
        heart_rate = st.number_input("Resting heart rate", 40, 150, 70)
        cholesterol_total = st.number_input("Total cholesterol", 100, 400, 200)
        hdl_cholesterol = st.number_input("HDL cholesterol", 20, 100, 50)
        ldl_cholesterol = st.number_input("LDL cholesterol", 50, 250, 120)
        triglycerides = st.number_input("Triglycerides", 50, 500, 150)
        smoking_status = st.selectbox("Smoker?", ["No", "Yes"])
        family_history = st.selectbox("Family history of diabetes?", ["No", "Yes"])
        hypertension_history = st.selectbox("History of hypertension?", ["No", "Yes"])
        cardiovascular_history = st.selectbox("History of cardiovascular disease?", ["No", "Yes"])
        ethnicity = st.selectbox("Ethnicity", ["Group A", "Group B", "Group C"])
        education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Higher"])
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Other"])
    
    submit_btn = st.form_submit_button("🩺 Check My Risk Now")
    
    if submit_btn:
        input_df = pd.DataFrame([{
            "age": age,
            "alcohol_consumption_per_week": alcohol,
            "physical_activity_minutes_per_week": physical_activity,
            "diet_score": diet_score,
            "sleep_hours_per_day": sleep_hours,
            "screen_time_hours_per_day": screen_time,
            "bmi": bmi,
            "waist_to_hip_ratio": waist_to_hip,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "cholesterol_total": cholesterol_total,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "triglycerides": triglycerides,
            "gender": 1 if gender=="Male" else 0,
            "ethnicity": ethnicity,
            "education_level": education_level,
            "income_level": income_level,
            "smoking_status": 1 if smoking_status=="Yes" else 0,
            "employment_status": employment_status,
            "family_history_diabetes": 1 if family_history=="Yes" else 0,
            "hypertension_history": 1 if hypertension_history=="Yes" else 0,
            "cardiovascular_history": 1 if cardiovascular_history=="Yes" else 0
        }])
        
        try:
            X_transformed = preprocessor.transform(input_df)
            prob = model.predict_proba(X_transformed)[0,1]
            pct = prob*100
            
            st.markdown("---")
            
            # --- Result Card ---
            if prob < 0.2:
                st.balloons()
                st.markdown(f"""
                <div class="result-card" style="background-color:#D5F5E3; border:2px solid #28B463;">
                    <h2 style="color:#196F3D;">🎉 Low Risk</h2>
                    <p class="big-font">Estimated Diabetes Risk: {pct:.1f}%</p>
                    <p style="color:#196F3D;">Maintain your healthy lifestyle!</p>
                </div>
                """, unsafe_allow_html=True)
            elif prob < 0.5:
                st.markdown(f"""
                <div class="result-card" style="background-color:#FCF3CF; border:2px solid #F1C40F;">
                    <h2 style="color:#9A7D0A;">⚠️ Moderate Risk</h2>
                    <p class="big-font">Estimated Diabetes Risk: {pct:.1f}%</p>
                    <p style="color:#9A7D0A;">Consider lifestyle improvements.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="background-color:#FADBD8; border:2px solid #E74C3C;">
                    <h2 style="color:#922B21;">🚨 High Risk</h2>
                    <p class="big-font">Estimated Diabetes Risk: {pct:.1f}%</p>
                    <p style="color:#922B21;">Consult a healthcare professional.</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")