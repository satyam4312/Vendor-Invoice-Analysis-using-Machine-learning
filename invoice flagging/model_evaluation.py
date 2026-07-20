from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train, X_test, y_test):

    rf = RandomForestClassifier(random_state = 42, n_jobs = -1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(rf, param_grid, scoring = scorer, cv = 5, n_jobs = -1, verbose = 2)

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    evaluate_model(best_model, X_test, y_test, "Random Forest Classifier")

    return best_model


def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    
    print(f"\n{model_name}")
    
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))













