import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


# ==========================
# Database Configuration
# ==========================
USERNAME = "root"
PASSWORD = quote_plus("YOUR_MYSQL_PASSWORD")
HOST = "localhost"
DATABASE = "ml_project"


# ==========================
# Load Data
# ==========================
def load_vendor_invoice_data():
    """
    Load vendor invoice data from MySQL.
    """
    engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}")

    query = """ SELECT Quantity, Dollars, Freight FROM vendor_invoice """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# ==========================
# Prepare Features
# ==========================
def prepare_features(df):
    """
    Prepare features and target variable.
    """
    # Keep only required columns
    df = df[["Quantity", "Dollars", "Freight", "Freight_per_unit"]].copy()

    # Remove missing values
    df = df.dropna()

    # Feature
    X = df[["Quantity", "Dollars", "Freight_per_unit"]]
    
    # Target
    y = df["Freight"]

    return X, y


# ==========================
# Split Dataset
# ==========================
def split_data(X, y, test_size = 0.2, random_state = 42):
    """
    Split dataset into training and testing sets.
    """
    return train_test_split(X, y, test_size = test_size, random_state = random_state)


# ==========================
# Test Pipeline
# ==========================
if __name__ == "__main__":

    df = load_vendor_invoice_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Total Records   : {len(df)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples : {len(X_test)}")
    
