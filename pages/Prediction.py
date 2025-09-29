import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🔮",
    layout='wide'
)

@st.cache_data
def setup_and_train_model():
    data = pd.read_csv('./dataset.csv')
    outlier_bloodPressure_index = data[data['resting bp s'] < 75].index
    outlier_Cholesterol_index = data[data['cholesterol'] == 0].index
    cleanedData = data.drop(outlier_bloodPressure_index,axis=0)
    cleanedData = data.drop(outlier_Cholesterol_index,axis=0)

    X = cleanedData.drop(['target'],axis=1)
    y = cleanedData['target']

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaled_X, y)
    
    return model,scaler

model,scaler = setup_and_train_model()

def detectDisease(input_data):
    df = pd.DataFrame([input_data])
    scaled_df = scaler.fit_transform(df)
    prediction = model.predict(scaled_df)
    return round(float(prediction[0]), 2)

st.title("💖 Heart Disease Prediction Model")
st.sidebar.success("1. Select your features\n2. Click 'Predict'")
st.sidebar.divider()
st.sidebar.header("Your Selections:")

col1, col2 = st.columns([1, 1.5])

with col2:
    st.header("Select the Features")
    
    age = st.slider('Age',20,80,40)
    sex = st.selectbox('Sex',['Female','Male'])
    chest_painType = st.selectbox('Chest Pain Type',['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'],1)
    resting_BPS = st.slider('Resting Blood Pressure',100,200,140)
    cholesterol = st.slider('Cholestrol',100,600,289)
    fasting_bloodSugar = st.selectbox('Fasting Blood Sugar',['False','True'])
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
        input_data['sex'] = ['Male','Female'].index(sex)
        input_data['chest pain type'] = ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'].index(chest_painType)+1
        input_data['fasting blood sugar'] = ['False','True'].index(fasting_bloodSugar)
        input_data['resting ecg'] = ['Normal','Wave Abnormality','Probable'].index(resting_ECG)
        input_data['exercise angina'] = ['False','True'].index(exerciseAngina)
        input_data['ST slope'] = ['Upward','Flat','Downward'].index(STSlope)+1

        disease = detectDisease(input_data)
        
        st.metric("Heart Disease", ["No Heart Disease","You have Heart Disease"][int(disease)])
    else:
        st.metric("Heart Disease", "No Heart Disease")