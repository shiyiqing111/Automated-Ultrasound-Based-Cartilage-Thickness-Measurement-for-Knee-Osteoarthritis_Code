import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def fit_predict_linear(X_train, Y_train, X_test, alpha=1.0):
    base = Ridge(alpha=alpha, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred, model
