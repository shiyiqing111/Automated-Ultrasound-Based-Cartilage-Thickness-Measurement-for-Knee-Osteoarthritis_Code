import numpy as np

def fit_predict_xgb(X_train, Y_train, X_test):
    try:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e:
        raise RuntimeError("The XGB model can be used only after installing xgboost") from e

    base = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred, model
