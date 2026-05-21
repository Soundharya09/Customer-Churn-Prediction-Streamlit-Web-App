import streamlit as st
import pandas as pd
import pickle

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ======================================
# LOAD MODEL FILES
# ======================================

with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# ======================================
# TITLE
# ======================================

st.title("📉 Customer Churn Prediction App")

st.write(
    "Predict whether a customer is likely to churn using Machine Learning."
)

# ======================================
# SIDEBAR INPUTS
# ======================================

st.sidebar.header("Customer Information")

# Numerical Inputs
age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=30
)

monthly_charge = st.sidebar.slider(
    "Monthly Charges",
    min_value=10,
    max_value=200,
    value=50
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    min_value=1,
    max_value=72,
    value=12
)

# Categorical Inputs
contract_type = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-Month", "One Year", "Two Year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber Optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic Check",
        "Mailed Check",
        "Bank Transfer",
        "Credit Card"
    ]
)

# ======================================
# CREATE INPUT DATAFRAME
# ======================================

input_data = {
    "Age": age,
    "MonthlyCharges": monthly_charge,
    "Tenure": tenure,
    "ContractType": contract_type,
    "InternetService": internet_service,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([input_data])

# ======================================
# ONE HOT ENCODING
# ======================================

input_encoded = pd.get_dummies(input_df)

# Match columns with training data
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# ======================================
# SCALE INPUT DATA
# ======================================

input_scaled = scaler.transform(input_encoded)

# ======================================
# PREDICTION
# ======================================

if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Customer is likely to churn.\n\n"
            f"Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay.\n\n"
            f"Probability of churn: {probability:.2%}"
        )

    # ======================================
    # CUSTOMER SUMMARY
    # ======================================

    st.subheader("Customer Summary")

    st.write(input_df)