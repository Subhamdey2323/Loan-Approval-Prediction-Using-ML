# Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be approved based on applicant information using machine learning. It includes data preprocessing, model training, evaluation, and a Streamlit app for interactive predictions.

---

## Project Overview

The goal of this project is to build a machine learning model that predicts loan approval (`Approved` or `Rejected`) based on applicant details such as age, income, employment experience, loan details, credit history, and other categorical features.

---

## Steps

### 1. Data Collection
- Dataset: `loan_data.csv`
- Contains applicant demographic details, loan details, credit information, and loan status.
- Example columns:
  - `person_age`, `person_gender`, `person_education`, `person_income`, `person_emp_exp`
  - `loan_amnt`, `loan_int_rate`, `loan_percent_income`
  - `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`
  - `person_home_ownership`, `loan_intent`
  - `loan_status` (target variable)

---

### 2. Data Exploration
- Check dataset structure using `df.head()`, `df.info()`, `df.describe()`.
- Check for missing values using `df.isnull().sum()`.
- Check duplicates using `df.duplicated().sum()`.
- Visualizations:
  - Count plots for categorical features vs loan status
  - Boxplots for numerical features vs loan status
  - Correlation heatmap to identify relationships between features

---

### 3. Data Preprocessing
- **Categorical Encoding**:
  - Binary mapping: `person_gender` (Male:1, Female:0), `previous_loan_defaults_on_file` (Yes:1, No:0)
  - Ordinal mapping: `person_education` (High School:0 → Doctorate:4)
  - One-hot encoding:
    - `person_home_ownership` → `home_OWN`, `home_RENT`, `home_OTHER`
    - `loan_intent` → `purpose_PERSONAL`, `purpose_EDUCATION`, etc.
- **Scaling**:
  - Features scaled using `StandardScaler`:
    ```
    ['person_age','person_gender','person_education','person_income','person_emp_exp',
     'loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length',
     'credit_score','previous_loan_defaults_on_file']
    ```

---

### 4. Model Training
- Models used:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Evaluation Metrics:
  - Accuracy, F1-score
  - Confusion matrix and classification report
- Best performing model (e.g., Random Forest) saved as `loan_approval_model.pkl`.
- Artifacts saved:
  - `scaler.pkl`
  - `feature_columns.pkl`

---

### 5. Streamlit Deployment
- Interactive web app (`app.py`) for predicting loan approval.
- Steps:
  1. User enters applicant details in the UI.
  2. Inputs are preprocessed:
     - Binary & ordinal mapping
     - One-hot encoding
     - Scaling numeric features
  3. Features aligned with `feature_columns`.
  4. Model predicts loan status and displays probability.
- Run the app:
  streamlit run app.py
