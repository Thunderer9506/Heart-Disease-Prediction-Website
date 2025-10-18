import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🔮",
    layout='wide'
)

@st.cache_resource
def setup_and_train_model():
    with open("model.pkl",'rb') as f:
        model = pickle.load(f)
    with open("evaluation.pkl",'rb') as f:
        evaluation = pickle.load(f)  

    return model, evaluation

model,evaluation = setup_and_train_model()

def detectDisease(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)

st.title("💖 Heart Disease Prediction Model")
st.sidebar.success("1. Select your features\n2. Click 'Predict'")
st.sidebar.divider()
st.sidebar.header("Your Selections:")

col1, col2 = st.columns([1, 1.5])

with col2:
    st.header("Select the Features")
    
    age = st.slider('Age',20,80,40)
    sex = st.selectbox('Sex',['Female','Male'],1)
    chest_painType = st.selectbox('Chest Pain Type',['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'],1)
    resting_BPS = st.slider('Resting Blood Pressure',100,200,140)
    cholesterol = st.slider('Cholestrol',100,600,289)
    fasting_bloodSugar = st.selectbox('Fasting Blood Sugar',['False','True'],0)
    resting_ECG = st.selectbox('Resting Electrocardiogram',['Normal','Wave Abnormality','Probable'])
    max_heartRate = st.slider('Max HeartRate',75,200,172)
    exerciseAngina = st.selectbox('Exercise Angina',['False','True'])
    oldPeak = st.slider('OldPeak',0,6)
    STSlope = st.selectbox('ST Slope',['Upward','Flat','Downward'])

selections = {
    "age":age,
    "sex":sex,
    "chest pain type":chest_painType,
    "resting bp s":resting_BPS,
    "cholesterol":cholesterol,
    "fasting blood sugar":fasting_bloodSugar,
    "resting ecg":resting_ECG,
    "max heart rate":max_heartRate,
    "exercise angina":exerciseAngina,
    "oldpeak":oldPeak,
    "ST slope":STSlope
}

for label, value in selections.items():
    st.sidebar.write(f"**{label}:** {value}")

with col1:
    st.header("Detect Heart Disease")
    
    if st.button("Predict 🔮", use_container_width=True):
        input_data = {
            "age":age,
            "sex":sex,
            "chest pain type":chest_painType,
            "resting bp s":resting_BPS,
            "cholesterol":cholesterol,
            "fasting blood sugar":fasting_bloodSugar,
            "resting ecg":resting_ECG,
            "max heart rate":max_heartRate,
            "exercise angina":exerciseAngina,
            "oldpeak":oldPeak,
            "ST slope":STSlope
        }
        input_data['sex'] = ['Female','Male'].index(sex)
        input_data['chest pain type'] = ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'].index(chest_painType)+1
        input_data['fasting blood sugar'] = ['False','True'].index(fasting_bloodSugar)
        input_data['resting ecg'] = ['Normal','Wave Abnormality','Probable'].index(resting_ECG)
        input_data['exercise angina'] = ['False','True'].index(exerciseAngina)
        input_data['ST slope'] = ['Upward','Flat','Downward'].index(STSlope)+1

        disease = detectDisease(input_data)
        
        st.metric("Heart Disease", ["No Heart Disease","You have Heart Disease"][int(disease)])
    else:
        st.metric("Heart Disease", "No Heart Disease")
    
    
    st.divider()
    
    st.subheader("Model Performance Metrics")
    
    met_col1, met_col2, met_col3 = st.columns(3)
    
    with met_col1:
        st.metric("Accuracy", f"{evaluation['Accuracy']:.2%}")
        st.metric("F1 Score", f"{evaluation['F1 Score']:.2%}")
    
    with met_col2:
        st.metric("Precision", f"{evaluation['Precision']:.2%}")
        st.metric("ROC AUC", f"{evaluation['ROC AUC']:.2%}")
    
    with met_col3:
        st.metric("Recall", f"{evaluation['Recall']:.2%}")
    
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Overall correct predictions (both positive and negative)
        - **Precision**: When model predicts heart disease, how often it is correct
        - **Recall**: Of all actual heart disease cases, how many were caught
        - **F1 Score**: Balance between precision and recall
        - **ROC AUC**: Model's ability to distinguish between classes (1.0 = perfect)
        
        Values closer to 100% indicate better performance.
        """)