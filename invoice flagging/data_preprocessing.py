import pandas as pd
import joblib
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ==========================
# Configuration
# ==========================
USERNAME = "root"
PASSWORD = quote_plus("satyam@1234#")
HOST = "localhost"
DATABASE = "ml_project"

RECEIVING_DELAY_THRESHOLD = 10
INVOICE_MISMATCH_THRESHOLD = 5


# ==========================
# Load Data
# ==========================
def load_invoice_data():
    """
    Load invoice and purchase data from MySQL.
    """

    engine = create_engine(
        f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    )

    query = """
    SELECT
        p.PONumber,

        p.total_brands,
        p.total_item_quantity,
        p.total_item_dollars,
        p.avg_receiving_delay,

        vi.invoice_quantity,
        vi.invoice_dollars,
        vi.Freight,
        vi.days_PO_to_invoice,
        vi.days_to_pay

    FROM
    (
        SELECT
            PONumber,
            COUNT(DISTINCT Brand) AS total_brands,
            SUM(Quantity) AS total_item_quantity,
            SUM(Dollars) AS total_item_dollars,
            AVG(DATEDIFF(ReceivingDate, PODate)) AS avg_receiving_delay

        FROM purchases

        GROUP BY PONumber
    ) p

    LEFT JOIN

    (
        SELECT
            PONumber,
            Quantity AS invoice_quantity,
            Dollars AS invoice_dollars,
            Freight,
            DATEDIFF(InvoiceDate, PODate) AS days_PO_to_invoice,
            DATEDIFF(PayDate, InvoiceDate) AS days_to_pay

        FROM vendor_invoice

    ) vi

    ON p.PONumber = vi.PONumber;
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# ==========================
# Create Labels
# ==========================
def create_invoice_risk_label(row):
    """
    Generate invoice fraud/risk label.
    """
    # Missing invoice record
    if pd.isna(row["invoice_dollars"]):
        return 1

    # Invoice amount mismatch
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > INVOICE_MISMATCH_THRESHOLD:
        return 1

    # Large receiving delay
    if row["avg_receiving_delay"] > RECEIVING_DELAY_THRESHOLD:
        return 1

    return 0


def apply_labels(df):
    """
    Apply risk labels to the dataframe.
    """
    df = df.copy()
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis = 1)
    return df


# ==========================
# Split Dataset
# ==========================
def split_data(df, features, target):
    """
    Split dataset into train and test sets.
    """
    X = df[features].copy()
    y = df[target].copy()

    return train_test_split(X, y, test_size = 0.20, random_state = 42)


# ==========================
# Scale Features
# ==========================
def scale_features(X_train, X_test, scaler_path = "models/scaler.pkl"):
    """
    Standardize numerical features and save scaler.
    """
    # Fill missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_test_scaled


# ==========================
# Complete Pipeline
# ==========================
if __name__ == "__main__":
    df = load_invoice_data()
    df = apply_labels(df)

    feature_columns = [
        "total_item_quantity",
        "total_item_dollars",
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
    ]

    target_column = "flag_invoice"

    X_train, X_test, y_train, y_test = split_data(df, feature_columns, target_column)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    print("Training samples :", len(X_train_scaled))
    print("Testing samples  :", len(X_test_scaled))
    print("Pipeline completed successfully.")















