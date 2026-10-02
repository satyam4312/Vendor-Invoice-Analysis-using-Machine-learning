import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Random Forest Regression
def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators = 200,
        random_state = 42,
        n_jobs = -1,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1
    )
    model.fit(X_train, y_train)
    return model


# Model Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Model" : model_name,
        "MAE" : mae,
        "RMSE" : rmse,
        "R2" : r2
    }









