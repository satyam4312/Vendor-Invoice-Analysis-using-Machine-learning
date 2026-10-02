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
    # 1. Create models directory
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load data
    print("Loading vendor invoice data...")
    df = load_vendor_invoice_data()
    print(f"Dataset shape: {df.shape}")

    # 3. Prepare features
    print("\nPreparing features...")
    X, y = prepare_features(df)
    print(f"Feature shape: {X.shape}")
    print(f"Target shape : {y.shape}")

    # 4. Split dataset
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # 5. Train all models
    print("\nTraining models...")
    print("-" * 60)

    models = {}

    # Linear Regression
    print("Training Linear Regression...")
    models["Linear Regression"] = train_linear_regression(X_train, y_train)

    # Decision Tree
    print("Training Decision Tree...")
    models["Decision Tree Regression"] = train_decision_tree(X_train, y_train)

    # Random Forest
    print("Training Random Forest...")
    models["Random Forest Regression"] = train_random_forest(X_train, y_train)

    # 6. Evaluate all models
    print("\nEvaluating models...")
    print("-" * 60)
    results = []

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

   
    # 7. Display comparison
    print("\nMODEL COMPARISON")
    print(
        f"{'Model':<30}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'R²':>12}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['Model']:<30}"
            f"{result['MAE']:>12.4f}"
            f"{result['RMSE']:>12.4f}"
            f"{result['R2']:>12.4f}"
        )

    # 8. Select best model based on MAE
    best_result = min(results, key=lambda result: result["MAE"])
    best_model_name = best_result["Model"]
    best_model = models[best_model_name]

    # 9. Save best model
    model_path = model_dir / "predict_freight_model1.pkl"
    joblib.dump(best_model, model_path)

    # 10. Final output
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Best Model : {best_model_name}")
    print(f"Best MAE   : {best_result['MAE']:.4f}")
    print(f"Best RMSE  : {best_result['RMSE']:.4f}")
    print(f"Best R²    : {best_result['R2']:.4f}")

    print(f"\nModel saved to:")
    print(f"  {model_path}")

    print("=" * 60)



if __name__ == "__main__":
    main()