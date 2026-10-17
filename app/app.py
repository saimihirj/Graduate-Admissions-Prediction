"""Exploration app for the Graduate Admissions dataset, with the prediction model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

from model_utils import (ELIGIBILITY_THRESHOLD, IMAGES, load_dataset,
                         predict_by_university, train_model)

st.set_page_config(page_title="Graduate Admissions Analysis",
                   page_icon=str(IMAGES / "logo.png"), layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def get_data():
    df = load_dataset("gad.csv").set_index("Serial No.")
    df.columns = [c.lower() for c in df.columns]
    return df


@st.cache_resource
def get_model():
    return train_model()


st.markdown("# Graduate Admissions Dataset")
st.markdown("### Analysis of the Graduate Admissions dataset")
st.image(Image.open(IMAGES / "gad.png"), width=720, caption="Graduate Admissions dataset")

st.markdown("### About the dataset")
st.info("The dataset describes the chance of admission of Indian students applying to "
        "graduate programmes, based on their test scores and profile. It was built to help "
        "students shortlist universities, and is inspired by the UCLA Graduate Dataset on Kaggle.")
st.image(Image.open(IMAGES / "univ.png"), width=720, caption="Top 5 universities in the US")

st.markdown("##### Attributes")
st.info("1. GRE score (out of 340)  \n2. TOEFL score (out of 120)  \n"
        "3. University rating (out of 5)  \n4. Statement of purpose, SOP (out of 5)  \n"
        "5. Letter of recommendation, LOR (out of 5)  \n6. Research experience (0 or 1)  \n"
        "7. CGPA (out of 10)  \n8. Chance of admission (0 to 1)")
st.image(Image.open(IMAGES / "par.png"), width=720, caption="Influence of each attribute")

df = get_data()

st.header("Quick explore")
st.sidebar.subheader("Quick explore")
st.markdown("Tick a box in the side panel to explore the dataset.")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(df)
if st.sidebar.checkbox("First rows"):
    st.subheader("First rows")
    st.write(df.head())
if st.sidebar.checkbox("Columns"):
    st.subheader("Columns")
    st.write(df.columns.to_list())
if st.sidebar.checkbox("Summary statistics"):
    st.subheader("Summary statistics")
    st.write(df.describe())
if st.sidebar.checkbox("Missing values"):
    st.subheader("Missing values")
    st.write(df.isnull().sum())

st.header("Create your own chart")
st.markdown("Tick a box in the side panel to draw a chart.")
st.sidebar.subheader("Create your own chart")
if st.sidebar.checkbox("Count plot"):
    st.subheader("Count plot")
    column = st.sidebar.selectbox("Column to count", df.columns[:5])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x=column, data=df, ax=ax)
    st.pyplot(fig)
if st.sidebar.checkbox("Distribution plot"):
    st.subheader("Distribution plot")
    column = st.sidebar.selectbox("Column to plot", df.columns[:5], key="dist")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

st.header("Prediction model")
st.sidebar.subheader("Prediction model")
st.markdown("Tick the box in the side panel to try the prediction model.")
if st.sidebar.checkbox("Show prediction model"):
    model, holdout_r2 = get_model()
    st.caption(f"Linear regression on all eight attributes; R-squared on a 20% hold-out set: {holdout_r2:.2f}.")
    gre = st.slider("GRE score (out of 340)", 260, 340, 316)
    toefl = st.slider("TOEFL score (out of 120)", 0, 120, 107)
    sop = st.slider("Statement of purpose (out of 5)", 1.0, 5.0, 3.5, step=0.5)
    lor = st.slider("Letters of recommendation (out of 5)", 1.0, 5.0, 3.5, step=0.5)
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, step=0.01)
    research = st.selectbox("Research experience", ("Yes", "No")) == "Yes"
    if st.button("Predict"):
        ielts, chances = predict_by_university(model, gre, toefl, sop, lor, cgpa, research)
        st.success(f"GRE {gre} · TOEFL {toefl} (about IELTS {ielts}) · CGPA {cgpa:.2f}")
        for rating, chance in enumerate(chances, start=1):
            st.info(f"University rating {rating} of 5: {chance:.0%} chance of admission")
            if chance >= ELIGIBILITY_THRESHOLD:
                st.success("A realistic option for your profile.")
            else:
                st.caption("A stretch for your profile.")

st.sidebar.subheader("Data source")
st.sidebar.info("[Graduate Admissions dataset on Kaggle](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)")
st.sidebar.subheader("Authors")
st.sidebar.info("[Sai Mihir Jakkaraju](https://github.com/saimihirj)  \n"
                "[Rushab Prakash Kulkarni](https://github.com/rushab14)")
