import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model, scaler, and feature columns
# -----------------------------
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# -----------------------------
# App Title
# -----------------------------
st.title("Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status.")

# -----------------------------
# User Inputs
# -----------------------------
person_age = st.number_input("Age", min_value=18)
person_gender = st.selectbox("Gender", ["Male", "Female"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Income", min_value=0)
person_emp_exp = st.number_input("Employment Experience (years)", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.01)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0)
credit_score = st.number_input("Credit Score", min_value=0)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "OTHER"])
loan_purpose = st.selectbox("Loan Purpose", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT"])

# -----------------------------
# Preprocess Inputs
# -----------------------------
# Binary mapping
person_gender = 1 if person_gender=="Male" else 0
previous_loan_defaults_on_file = 1 if previous_loan_defaults_on_file=="Yes" else 0

# Ordinal mapping
education_map = {'High School':0, 'Associate':1, 'Bachelor':2, 'Master':3, 'Doctorate':4}
person_education = education_map[person_education]

# One-hot encoding for Home Ownership
home_dict = {'home_OTHER':0, 'home_OWN':0, 'home_RENT':0}
home_dict[f'home_{home_ownership}'] = 1

# One-hot encoding for Loan Purpose
purpose_dict = {'purpose_EDUCATION':0, 'purpose_HOMEIMPROVEMENT':0, 
                'purpose_MEDICAL':0, 'purpose_PERSONAL':0, 'purpose_VENTURE':0}
purpose_dict[f'purpose_{loan_purpose}'] = 1

# Combine all inputs into a single DataFrame
input_df = pd.DataFrame([{
    'person_age': person_age,
    'person_gender': person_gender,
    'person_education': person_education,
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
    **home_dict,
    **purpose_dict
}])

# -----------------------------
# Scale the 11 columns that were used during training
# -----------------------------
scaler_columns = ['person_age','person_gender','person_education','person_income','person_emp_exp',
                  'loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length',
                  'credit_score','previous_loan_defaults_on_file']

input_df[scaler_columns] = scaler.transform(input_df[scaler_columns])

# -----------------------------
# Align all columns with training
# -----------------------------
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Loan Status"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    st.write(f"Probability of Loan Approval: {pred_proba*100:.2f}%")
    if pred == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
