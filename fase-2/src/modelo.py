"""Separacion de datos, construccion del pipeline, entrenamiento y evaluacion."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocesamiento import TransformadorCasas


def separar_datos(df, objetivo, test_size, semilla):
    """Separa entrenamiento y prueba ANTES de ajustar cualquier transformacion."""
    X = df.drop(columns=[objetivo])
    y = df[objetivo].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=semilla)


def excluir_por_id(X, y, columna_id, ids):
    """Quita observaciones atipicas, solo del conjunto de entrenamiento."""
    if not ids or columna_id not in X.columns:
        return X, y, []
    mascara = X[columna_id].isin(ids)
    return X[~mascara], y[~mascara], X.loc[mascara, columna_id].tolist()


def construir_pipeline(alpha, max_iter):
    return Pipeline([
        ("transformador", TransformadorCasas()),
        ("modelo", Lasso(alpha=alpha, max_iter=max_iter)),
    ])


def construir_modelo_base():
    return Pipeline([
        ("transformador", TransformadorCasas()),
        ("modelo", DummyRegressor(strategy="median")),
    ])


def calcular_metricas(y_real, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "R2": float(r2_score(y_real, y_pred)),
    }


def validacion_cruzada(pipeline, X, y, folds, semilla):
    """Cada fold reajusta el pipeline completo, sin fuga hacia su particion de validacion."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=semilla)
    r = cross_validate(pipeline, X, y, cv=kfold, scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    })
    signo = {"MAE": -1, "RMSE": -1, "R2": 1}
    return {
        m: {"media": float(s * r[f"test_{m}"].mean()), "desviacion": float(r[f"test_{m}"].std())}
        for m, s in signo.items()
    }
