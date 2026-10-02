import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# Page Configuration (must be the first Streamlit call)
# ==============================
st.set_page_config(
    page_title="Vendor Invoice Analytics",
    page_icon="📊",
    layout="wide",
)

# ==============================
# Styling
# ==============================
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
        padding: 2rem 2.2rem; border-radius: 16px; color: #fff; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
    .hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.05rem;}
    .feature {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px; padding: .9rem 1rem; text-align: center; font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);
        padding: 1rem 1.2rem; border-radius: 12px;
    }
    .stButton > button, .stFormSubmitButton > button {
        width: 100%; border-radius: 10px; font-weight: 600; padding: .6rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Constants
# ==============================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

INVOICE_COLS = [
    "total_item_quantity",
    "total_item_dollars",
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
]
FREIGHT_COLS = ["Quantity", "Dollars", "Freight_per_unit"]


# ==============================
# Load Models (with error handling)
# ==============================
@st.cache_resource
def load_models():
    invoice_model = joblib.load(MODEL_DIR / "predict_flag_invoice1.pkl")
    freight_model = joblib.load(MODEL_DIR / "predict_freight_model1.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler1.pkl")
    return invoice_model, freight_model, scaler


try:
    invoice_model, freight_model, scaler = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: `{e.filename}`. Make sure the `models/` folder sits next to `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the models: {e}")
    st.stop()


# ==============================
# Helpers
# ==============================
def prepare(df: pd.DataFrame, estimator, default_cols):
    """Select/order columns exactly as the fitted estimator expects."""
    cols = list(getattr(estimator, "feature_names_in_", default_cols))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return df[cols]


def run_invoice(df: pd.DataFrame):
    X = scaler.transform(prepare(df, scaler, INVOICE_COLS))
    preds = invoice_model.predict(X)
    probs = invoice_model.predict_proba(X)[:, 1] if hasattr(invoice_model, "predict_proba") else None
    return preds, probs


def run_freight(df: pd.DataFrame):
    return freight_model.predict(prepare(df, freight_model, FREIGHT_COLS))


def risk_level(p: float) -> str:
    return "Low" if p < 0.30 else "Medium" if p < 0.60 else "High"


def read_csv(uploaded, required):
    df = pd.read_csv(uploaded)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing column(s): {', '.join(missing)}")
    return df


# ==============================
# Sidebar
# ==============================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choose Module",
    ["🏠 Home", "📦 Freight Cost Prediction", "⚠️ Invoice Risk Detection"],
)
st.sidebar.divider()
st.sidebar.success("Models loaded ✔")
st.sidebar.caption("Predictions are decision support, not a substitute for invoice review.")

# ==============================
# Home
# ==============================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Vendor Invoice Intelligence Portal</h1>
            <p>AI-driven freight cost prediction &amp; invoice risk detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="feature">📦<br>Forecast Freight Costs</div>', unsafe_allow_html=True)
    c2.markdown('<div class="feature">⚠️<br>Detect Risky Invoices</div>', unsafe_allow_html=True)
    c3.markdown('<div class="feature">💰<br>Reduce Financial Leakage</div>', unsafe_allow_html=True)
    c4.markdown('<div class="feature">🤖<br>Automate Approvals</div>', unsafe_allow_html=True)
    st.write("")
    st.info("Pick a module from the sidebar. Each module supports a single prediction or a CSV batch upload.")

# ==============================
# Freight Cost Prediction
# ==============================
elif page == "📦 Freight Cost Prediction":
    st.header("📦 Freight Cost Prediction")
    st.caption("Estimate freight cost from Quantity, Dollars and Freight per unit.")

    tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV)"])

    with tab_single:
        with st.form("freight_form"):
            c1, c2, c3 = st.columns(3)
            quantity = c1.number_input("Quantity", min_value=1.0, value=10000.0, step=100.0)
            dollars = c2.number_input("Dollars ($)", min_value=1.0, value=100000.0, step=1000.0)
            freight_per_unit = c3.number_input("Freight per unit ($)", min_value=0.0, value=100.0, step=1.0)
            submit = st.form_submit_button("Predict Freight", type="primary")

        if submit:
            try:
                input_df = pd.DataFrame(
                    {"Quantity": [quantity], "Dollars": [dollars], "Freight_per_unit": [freight_per_unit]}
                )
                prediction = float(run_freight(input_df)[0])
                m1, m2 = st.columns(2)
                m1.metric("Estimated Freight Cost", f"${prediction:,.2f}")
                m2.metric("Freight as % of Dollars", f"{prediction / dollars:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_batch:
        st.caption(f"Required columns: {', '.join(FREIGHT_COLS)}")
        file = st.file_uploader("Upload CSV", type="csv", key="freight_csv")
        if file is not None:
            try:
                df = read_csv(file, FREIGHT_COLS)
                df["Predicted_Freight"] = run_freight(df)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇ Download results",
                    df.to_csv(index=False).encode("utf-8"),
                    "freight_predictions.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Could not process file: {e}")

# ==============================
# Invoice Risk Detection
# ==============================
else:
    st.header("⚠️ Invoice Risk Detection")
    st.caption("Flag invoices that look abnormal based on quantities, dollars and freight.")

    tab_single, tab_batch = st.tabs(["Single invoice", "Batch (CSV)"])

    with tab_single:
        with st.form("invoice_form"):
            c1, c2 = st.columns(2)
            with c1:
                total_item_quantity = st.number_input("Total Item Quantity", min_value=0.0, step=1.0)
                total_item_dollars = st.number_input("Total Item Dollars ($)", min_value=0.0, step=10.0)
                invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0, step=1.0)
            with c2:
                invoice_dollars = st.number_input("Invoice Dollars ($)", min_value=0.0, step=10.0)
                freight = st.number_input("Freight ($)", min_value=0.0, step=1.0)
            submit = st.form_submit_button("Predict Invoice Risk", type="primary")

        if submit:
            if invoice_quantity == 0 and invoice_dollars == 0:
                st.warning("Enter invoice quantity and dollars before predicting.")
            else:
                try:
                    df = pd.DataFrame(
                        {
                            "total_item_quantity": [total_item_quantity],
                            "total_item_dollars": [total_item_dollars],
                            "invoice_quantity": [invoice_quantity],
                            "invoice_dollars": [invoice_dollars],
                            "Freight": [freight],
                        }
                    )
                    preds, probs = run_invoice(df)

                    if preds[0] == 1:
                        st.error("⚠️ High Risk Invoice — recommend manual review")
                    else:
                        st.success("✅ Normal Invoice — safe to approve")

                    if probs is not None:
                        p = float(probs[0])
                        m1, m2 = st.columns(2)
                        m1.metric("Risk Probability", f"{p:.2%}")
                        m2.metric("Risk Level", risk_level(p))
                        st.progress(min(max(p, 0.0), 1.0))
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with tab_batch:
        st.caption(f"Required columns: {', '.join(INVOICE_COLS)}")
        file = st.file_uploader("Upload CSV", type="csv", key="invoice_csv")
        if file is not None:
            try:
                df = read_csv(file, INVOICE_COLS)
                preds, probs = run_invoice(df)
                df["Risk_Flag"] = ["High Risk" if p == 1 else "Normal" for p in preds]
                if probs is not None:
                    df["Risk_Probability"] = probs
                k1, k2 = st.columns(2)
                k1.metric("Invoices scored", f"{len(df):,}")
                k2.metric("Flagged high risk", f"{int((preds == 1).sum()):,}")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇ Download results",
                    df.to_csv(index=False).encode("utf-8"),
                    "invoice_risk_predictions.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Could not process file: {e}")