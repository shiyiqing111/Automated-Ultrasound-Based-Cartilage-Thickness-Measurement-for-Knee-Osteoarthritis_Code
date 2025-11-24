import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

def build_rf(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1):
    base = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    return MultiOutputRegressor(base)

def fit_predict_rf(X_train, Y_train, X_test):
    model = build_rf()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred, model
