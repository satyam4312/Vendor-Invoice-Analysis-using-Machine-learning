import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title = "Vendor Invoice Analytics",
    page_icon = "📊",
    layout = "wide"
)

st.title("📊 Vendor Invoice Analytics System")

st.markdown("""
# Vendor Invoice Intelligence Portal

### AI-Driven Freight Cost Prediction & Invoice Risk Detection

This application leverages Machine Learning to:

- 📦 Forecast Freight Costs
- ⚠ Detect Risky Vendor Invoices
- 💰 Reduce Financial Leakage
- 🤖 Automate Invoice Approval
""")

st.divider()

# ==============================
# Load Models
# ==============================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_models():

    invoice_model = joblib.load(MODEL_DIR / "predict_flag_invoice.pkl")
    freight_model = joblib.load(MODEL_DIR / "predict_freight_model2.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    return invoice_model, freight_model, scaler


invoice_model, freight_model, scaler = load_models()

# ==============================
# Sidebar
# ==============================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Module",
    [
        "Freight Cost Prediction",
        "Invoice Risk Detection"
    ]
)

# ==========================================================
# Invoice Risk Detection
# ==========================================================

if page == "Invoice Risk Detection":
    
    st.header("⚠ Invoice Risk Detection")
    
    col1, col2 = st.columns(2)

    with col1:
        total_item_quantity = st.number_input(
            "Total Item Quantity",
            min_value=0.0
        )

        total_item_dollars = st.number_input(
            "Total Item Dollars",
            min_value=0.0
        )

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0
        )

    with col2:

        invoice_dollars = st.number_input(
            "Invoice Dollars",
            min_value=0.0
        )

        freight = st.number_input(
            "Freight",
            min_value=0.0
        )

    if st.button("Predict Invoice Risk"):

        df = pd.DataFrame({
            "total_item_quantity":[total_item_quantity],
            "total_item_dollars":[total_item_dollars],
            "invoice_quantity":[invoice_quantity],
            "invoice_dollars":[invoice_dollars],
            "Freight":[freight]
        })

        df_scaled = scaler.transform(df)
        prediction = invoice_model.predict(df_scaled)[0]
        probability = invoice_model.predict_proba(df_scaled)[0][1]

        st.subheader("Prediction")

        if prediction == 1:
            st.error("⚠ High Risk Invoice")
        else:
            st.success("✅ Normal Invoice")

        st.metric(
            "Risk Probability",
            f"{probability:.2%}"
        )

# ==========================================================
# Freight Prediction
# ==========================================================

else:

    st.header("📦 Freight Cost Prediction")

    st.write("Predict the estimated freight cost using Quantity and Invoice Dollars.")

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1.0,
                value=1000.0
            )

        with col2:
            dollars = st.number_input(
                "Dollars",
                min_value=1.0,
                value=10000.0
            )

        submit = st.form_submit_button("Predict Freight")

    if submit:
        input_df = pd.DataFrame({
            "Quantity":[quantity],
            "Dollars":[dollars]
        })

        prediction = freight_model.predict(input_df)[0]

        st.success("Prediction completed successfully.")

        st.metric(
            label="Estimated Freight Cost",
            value=f"${prediction:,.2f}"
        )
