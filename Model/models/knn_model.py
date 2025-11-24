import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

def build_knn(n_neighbors=5, weights='distance'):
    base = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    return MultiOutputRegressor(base)

def fit_predict_knn(X_train, Y_train, X_test, save_model_path=None):
    model = build_knn()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    if save_model_path:
        joblib.dump(model, save_model_path)
    return Y_pred, model
