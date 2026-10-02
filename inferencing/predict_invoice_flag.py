import os
import joblib
import pandas as pd

MODEL_PATH = "models/predict_flag_invoice1.pkl"
SCALER_PATH = "models/scaler1.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_invoice_flag(input_data):
    model, scaler = load_model()
    input_df = pd.DataFrame(input_data)

    feature_order = [
        "total_item_quantity",
        "total_item_dollars",
        "invoice_quantity",
        "invoice_dollars",
        "Freight"
    ]

    input_df = input_df[feature_order]

    # Scale input
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    result = input_df.copy()

    result["Predicted_Flag"] = prediction
    result["Risk_Probability"] = probability[:, 1]

    return result


if __name__ == "__main__":

    sample_data = [{
        "total_item_quantity":150,
        "total_item_dollars":3500,
        "invoice_quantity":150,
        "invoice_dollars":3520,
        "Freight":120
    }]

    print(predict_invoice_flag(sample_data))
    