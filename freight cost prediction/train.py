import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    # Create models directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok = True)

    # 1. Load data
    df = load_vendor_invoice_data()

    # 2. Prepare features
    X, y = prepare_features(df)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train models
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree Regression": train_decision_tree(X_train, y_train),
        "Random Forest Regression": train_random_forest(X_train, y_train)
    }

    # 5. Evaluate models
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    # 6. Display results
    print("Model Evaluation Results")
    print("-" * 50)

    for result in results:
        print(
            f"{result['Model']}: "
            f"MAE = {result['MAE']:.4f}, "
            f"RMSE = {result['RMSE']:.4f}, "
            f"R² = {result['R2']:.4f}"
        )

    # 7. Select best model based on MAE
    best_result = min(results, key=lambda r: r["MAE"])
    best_model_name = best_result["Model"]
    best_model = models[best_model_name]


    # 8. Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    # 9. Final output
    print("\nTraining completed successfully.")
    print(f"Best Model : {best_model_name}")
    print(f"Best MAE   : {best_result['MAE']:.4f}")
    print(f"Model Saved: {model_path}")


if __name__ == "__main__":
    main()
    