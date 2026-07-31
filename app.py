import streamlit as st
import pandas as pd
import joblib

# ---------------- Load Model ----------------
model = joblib.load("decision_tree_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below.")

# ---------------- User Input ----------------

gender = st.selectbox("Gender", ["Male", "Female"])

married = st.selectbox("Married", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

education = st.selectbox("Education", ["Graduate", "Not Graduate"])

self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150.0
)

loan_term = st.selectbox(
    "Loan Amount Term",
    [12,36,60,84,120,180,240,300,360,480]
)

credit_history = st.selectbox(
    "Credit History",
    [1.0,0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban","Semiurban","Rural"]
)

# ---------------- Prediction ----------------

if st.button("Predict Loan Status"):

    input_df = pd.DataFrame({
        "Gender":[gender],
        "Married":[married],
        "Dependents":[dependents],
        "Education":[education],
        "Self_Employed":[self_employed],
        "ApplicantIncome":[applicant_income],
        "CoapplicantIncome":[coapplicant_income],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[loan_term],
        "Credit_History":[credit_history],
        "Property_Area":[property_area]
    })

    # Label Encoding
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)

    result = target_encoder.inverse_transform(prediction)

    st.subheader("Prediction")

    if result[0] == "Y":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")