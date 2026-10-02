import os
import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model_1.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load the trained freight cost prediction model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def predict_freight_cost(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['predicted_freight'] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":
    sample_data = [
        {
            "Quantity" : 1935,
            "Dollars" : 15527.25,
            "Freight_per_unit" : 0.221809,
        }
    ]

    result = predict_freight_cost(sample_data)
    print(result) 