import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Home", page_icon="👋", layout="wide")

@st.cache_data
def load_data(path="dataset.csv"):
    return pd.read_csv(path)

data = load_data()

# Sidebar - quick controls
st.sidebar.success("Go to Prediction Page to Predict If you have Heart Disease or not")

# Header
st.title("💖 Heart Disease — Data Overview")
st.markdown(
    "Interactive exploratory page. Use the sidebar to filter samples and view raw data or download cleaned data."
)

# Data cleaning (fixed: drop both outlier sets from original df)
def cleaningData(df: pd.DataFrame) -> pd.DataFrame:
    outlier_bp_idx = df[df['resting bp s'] < 75].index
    outlier_chol_idx = df[df['cholesterol'] == 0].index
    to_drop = outlier_bp_idx.union(outlier_chol_idx)
    cleaned = df.drop(index=to_drop).reset_index(drop=True)
    return cleaned

cleaned = cleaningData(data)
target_counts = cleaned['target'].value_counts()
have_hd = cleaned[cleaned['target'] == 1]

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows (cleaned)", f"{len(cleaned):,}")
col2.metric("Heart disease (%)", f"{100 * target_counts.get(1, 0) / len(cleaned):.1f}%")
col3.metric("Mean age", f"{cleaned['age'].mean():.1f}")
col4.metric("Male (%)", f"{100 * (cleaned['sex'] == 1).mean():.0f}%")

# Expanders for quick data preview + info
with st.expander("Dataset head & download"):
    st.write(cleaned.head())
    csv = cleaned.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", data=csv, file_name="cleaned_dataset.csv", mime="text/csv")

with st.expander("Dataset info & description"):
    buffer = io.StringIO()
    cleaned.info(buf=buffer)
    st.text(buffer.getvalue())
    st.write(cleaned.describe())

tabs = st.tabs([
    "Overview & Results",
    "Continuous Features (all)",
    "Categorical Features (all)",
    "Continuous (have heart disease)",
    "Categorical (have heart disease)",
    "Correlation Matrix"
])

# Tab 0: Overview & Results (include the commented results here)
with tabs[0]:
    st.header("Overview")
    st.write("Quick dataset snapshot")
    st.dataframe(cleaned.drop(['target'],axis=1).head(10))

    st.subheader("Results / Observations")
    st.markdown(
        """
        1. People having heart disease generally are of 50-60 age.  
        2. Males are more prone to heart disease compared to females.  
        3. Heart disease generally occurs with Type 4 chest pain (asymptomatic).  
        4. Heart disease is more common in people who are not fasting blood sugar.  
        5. Heart disease patients generally show exercise angina (chest pain due to lack of oxygen).
        """
    )

# Tab 1: Continuous features (all data)
with tabs[1]:
    st.header("Continuous Features — All samples")
    cont_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    sel_cont = st.multiselect("Select continuous features to plot", cont_cols, default=cont_cols)
    if sel_cont:
        fig, axs = plt.subplots(nrows=max(1, (len(sel_cont)+2)//3), ncols=3, figsize=(15, 4*((len(sel_cont)+2)//3)))
        axs = axs.flatten()
        for ax in axs[len(sel_cont):]:
            ax.set_visible(False)
        for i, col in enumerate(sel_cont):
            sns.histplot(cleaned, x=col, hue='target', multiple='stack', kde=True, ax=axs[i])
            axs[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least one continuous feature to plot.")

# Tab 2: Categorical features (all data)
with tabs[2]:
    st.header("Categorical Features — All samples")
    cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
    sel_cat = st.multiselect("Select categorical features to plot", cat_cols, default=cat_cols[:4])
    if sel_cat:
        fig, axs = plt.subplots(nrows=max(1, (len(sel_cat)+2)//3), ncols=3, figsize=(15, 4*((len(sel_cat)+2)//3)))
        axs = axs.flatten()
        for ax in axs[len(sel_cat):]:
            ax.set_visible(False)
        for i, col in enumerate(sel_cat):
            sns.countplot(data=cleaned, x=col, hue='target', ax=axs[i])
            axs[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least one categorical feature to plot.")

# Tab 3: Continuous features for people having heart disease
with tabs[3]:
    st.header("Continuous Features — People with Heart Disease")
    cont_cols_hd = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    sel_cont_hd = st.multiselect("Select continuous features to plot (HD)", cont_cols_hd, default=cont_cols_hd)
    if sel_cont_hd:
        fig, axs = plt.subplots(nrows=max(1, (len(sel_cont_hd)+2)//3), ncols=3, figsize=(15, 4*((len(sel_cont_hd)+2)//3)))
        axs = axs.flatten()
        for ax in axs[len(sel_cont_hd):]:
            ax.set_visible(False)
        for i, col in enumerate(sel_cont_hd):
            sns.histplot(have_hd, x=col, kde=True, ax=axs[i], color='crimson')
            axs[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least one continuous feature to plot.")

# Tab 4: Categorical features for people having heart disease
with tabs[4]:
    st.header("Categorical Features — People with Heart Disease")
    cat_cols_hd = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']
    sel_cat_hd = st.multiselect("Select categorical features to plot (HD)", cat_cols_hd, default=cat_cols_hd[:4])
    if sel_cat_hd:
        fig, axs = plt.subplots(nrows=max(1, (len(sel_cat_hd)+2)//3), ncols=3, figsize=(15, 4*((len(sel_cat_hd)+2)//3)))
        axs = axs.flatten()
        for ax in axs[len(sel_cat_hd):]:
            ax.set_visible(False)
        for i, col in enumerate(sel_cat_hd):
            sns.countplot(data=have_hd, x=col, ax=axs[i], color='crimson')
            axs[i].set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least one categorical feature to plot.")

# Tab 5: Correlation matrix
with tabs[5]:
    st.header("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cleaned.corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax)
    st.pyplot(fig)