from model_evaluation import train_random_forest, evaluate_model
import joblib
from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data,
    scale_features
)


features = ['total_item_quantity', 'total_item_dollars', 
           'invoice_quantity', 'invoice_dollars',	
           'Freight'] 

target = "flag_invoice"


def main():
    # load data
    df = load_invoice_data()
    df = apply_labels(df)

    # prepare data
    X_train, X_test, y_train, y_test = split_data(df, features, target)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, "models/scaler.pkl")

    # train and evaluate model
    best_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    evaluate_model(best_model, X_test_scaled, y_test, "Random Forest Classifier")

    # save best model
    joblib.dump(best_model, "models/predict_flag_invoice.pkl")


if __name__ == "__main__":
    main()