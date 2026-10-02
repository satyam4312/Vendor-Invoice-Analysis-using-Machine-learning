import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_random_forest,
    evaluate_model
)


def main():
    # ==========================================================
    # 1. Create models directory
    # ==========================================================

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)


    # ==========================================================
    # 2. Load data
    # ==========================================================

    print("Loading vendor invoice data...")
    df = load_vendor_invoice_data()
    print(f"Dataset shape: {df.shape}")


    # ==========================================================
    # 3. Prepare features
    # ==========================================================

    print("\nPreparing features...")
    X, y = prepare_features(df)
    print(f"Feature shape: {X.shape}")
    print(f"Target shape : {y.shape}")


    # ==========================================================
    # 4. Split dataset
    # ==========================================================

    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")


    # ==========================================================
    # 5. Train Random Forest
    # ==========================================================

    print("\nTraining Random Forest Regression...")
    model = train_random_forest(X_train, y_train)


    # ==========================================================
    # 6. Evaluate model
    # ==========================================================

    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)


    # ==========================================================
    # 7. Display results
    # ==========================================================

    print("\n" + "=" * 60)
    print("RANDOM FOREST REGRESSION RESULTS")
    print("=" * 60)

    print(f"MAE  : {results['MAE']:.4f}")
    print(f"RMSE : {results['RMSE']:.4f}")
    print(f"R²   : {results['R2']:.4f}")


    # ==========================================================
    # 8. Save model
    # ==========================================================

    model_path = model_dir / "predict_freight_model1.pkl"
    joblib.dump(model, model_path)


    # ==========================================================
    # 9. Final output
    # ==========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Model : Random Forest Regression")
    print(f"MAE   : {results['MAE']:.4f}")
    print(f"RMSE  : {results['RMSE']:.4f}")
    print(f"R²    : {results['R2']:.4f}")
    print(f"Saved : {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()