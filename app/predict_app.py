"""Prediction-only app: estimates the chance of admission for each university rating."""
import streamlit as st

from model_utils import ELIGIBILITY_THRESHOLD, IMAGES, predict_by_university, train_model

st.set_page_config(page_title="Graduate Admission Prediction",
                   page_icon=str(IMAGES / "logo.png"), layout="wide")


@st.cache_resource
def get_model():
    return train_model()


def main():
    st.title("Graduate Admissions Prediction")
    st.caption("Estimated chance of admission at universities of each rating, "
               "from a linear regression trained on the Graduate Admissions dataset.")

    gre = st.slider("GRE score (out of 340)", 260, 340, 316)
    toefl = st.slider("TOEFL score (out of 120)", 0, 120, 107)
    sop = st.slider("Statement of purpose (out of 5)", 1.0, 5.0, 3.5, step=0.5)
    lor = st.slider("Letters of recommendation (out of 5)", 1.0, 5.0, 3.5, step=0.5)
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, step=0.01)
    research = st.selectbox("Research experience", ("Yes", "No")) == "Yes"

    if st.button("Predict"):
        model, _ = get_model()
        ielts, chances = predict_by_university(model, gre, toefl, sop, lor, cgpa, research)
        st.success(f"GRE {gre} · TOEFL {toefl} (about IELTS {ielts}) · CGPA {cgpa:.2f}")
        for rating, chance in enumerate(chances, start=1):
            st.info(f"University rating {rating} of 5: {chance:.0%} chance of admission")
            if chance >= ELIGIBILITY_THRESHOLD:
                st.success("A realistic option for your profile.")
            else:
                st.caption("A stretch for your profile.")


if __name__ == "__main__":
    main()
