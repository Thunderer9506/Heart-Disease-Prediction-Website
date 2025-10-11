import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.sidebar.success("Go to Prediction Page to Predict If you have Heart Disease or not")

data = pd.read_csv('dataset.csv')

st.title("Heart Disease Prediction Model")

st.write("## Statistics")

st.write('### Shape of the data: ')
st.write(f'Rows : {data.shape[0]}')
st.write(f'Cols( {data.shape[1]} ) : `{", ".join(list(data.columns))}`')

st.divider()

st.write('### Data Head')
st.write(data.head())

buffer = io.StringIO()
data.info(buf=buffer)
info = buffer.getvalue()

st.divider()

st.write('### Data Info')
st.text(info[37:])

st.divider()

st.write('### Data Described')
st.write(data.describe())

st.divider()

def cleaningData():
    outlier_bloodPressure_index = data[data['resting bp s'] < 75].index
    outlier_Cholesterol_index = data[data['cholesterol'] == 0].index
    cleanedData = data.drop(outlier_bloodPressure_index,axis=0)
    cleanedData = data.drop(outlier_Cholesterol_index,axis=0)
    return cleanedData

cleanedData = cleaningData()
data_of_people_have_heart_disease = cleanedData[cleanedData['target'] == 1]

st.write('## Exploratory Data Analysis')

st.write('### Continues Features Distibution')

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
distributedColumns = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

for col, ax in zip(distributedColumns, axs.flatten()):
    sns.histplot(cleanedData, x=col, kde=True, ax=ax)
    ax.set_xlabel(col)

# hide the unused subplot (last one)
axs.flatten()[-1].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.divider()

st.write('### Cateogorial Distributions')

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
categoryColumns = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

for col, ax in zip(categoryColumns, axs.flatten()):
    ax = sns.countplot(cleanedData, x=col, ax=ax)
    ax.set_xlabel(col.title())

plt.tight_layout()

st.pyplot(fig)

st.divider()

st.write('### Continues Features Distibution of people having heart disease')

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
distributedColumns = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

for col, ax in zip(distributedColumns, axs.flatten()):
    sns.histplot(data_of_people_have_heart_disease, x=col, kde=True, ax=ax)
    ax.set_xlabel(col)

# hide the unused subplot (last one)
axs.flatten()[-1].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.divider()

st.write('### Cateogorial Distributions of people having heart disease')

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
categoryColumns = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

for col, ax in zip(categoryColumns, axs.flatten()):
    ax = sns.countplot(data_of_people_have_heart_disease, x=col, ax=ax)
    ax.set_xlabel(col.title())

plt.tight_layout()

st.pyplot(fig)

st.divider()


st.write("## Results")
st.text('1. People having heart disease generally are of 50-60 age')
st.text('2. Males are more prone to heart disease compare to females')
st.text('3. Heart disease generally occur due to Type 4 chest pain( asymptomatic )')
st.text('4. Heart disease is prone to those people who are not fasting blood sugar')
st.text('5. Heart disease patient generally see a exercise anigna( chest pain due to lack of oxygen )')