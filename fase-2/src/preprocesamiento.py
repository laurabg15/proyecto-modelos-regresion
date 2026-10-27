"""Transformaciones de datos del modelo de precios de viviendas (Ames).

Es la unica definicion del preprocesamiento: la usan train.py y predict.py
a traves del Pipeline guardado, por lo que no se duplica entre scripts.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

ESCALA_CALIDAD = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# Columnas de entrada que necesita el modelo y su tipo esperado
ESQUEMA_ENTRADA = {
    "OverallQual": "numerica",
    "ExterQual": "categorica",
    "BsmtQual": "categorica",
    "TotalBsmtSF": "numerica",
    "CentralAir": "categorica",
    "1stFlrSF": "numerica",
    "GrLivArea": "numerica",
    "KitchenQual": "categorica",
    "GarageFinish": "categorica",
    "GarageArea": "numerica",
}


class TransformadorCasas(BaseEstimator, TransformerMixin):
    FEATURES_ORIGINALES = list(ESQUEMA_ENTRADA)
    QUAL_COLS = ["ExterQual", "BsmtQual", "KitchenQual"]
    # 0 = la instalacion no existe (p. ej. vivienda sin sotano)
    QUAL_MAPPING = ESCALA_CALIDAD
    COLS_A_ESCALAR = ["TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea"]

    def _validar_columnas(self, X):
        faltantes = [c for c in self.FEATURES_ORIGINALES if c not in X.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    def _imputar(self, df):
        # En Ames, un NaN en sotano/garaje significa que la vivienda no lo tiene
        df["BsmtQual"] = df["BsmtQual"].fillna(0)
        df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)
        df["GarageArea"] = df["GarageArea"].fillna(0)
        df["GarageFinish"] = df["GarageFinish"].fillna("SinGaraje")
        # Para el resto se usa el valor aprendido en entrenamiento
        for c, valor in self.valores_imputacion_.items():
            df[c] = df[c].fillna(valor)
        return df

    def _preparar(self, X):
        self._validar_columnas(X)
        df = X[self.FEATURES_ORIGINALES].copy()
        for c in self.QUAL_COLS:
            df[c] = df[c].map(self.QUAL_MAPPING)
        df["GarageFinish"] = df["GarageFinish"].replace("RFn", "Fin")
        return df

    def fit(self, X, y=None):
        df = self._preparar(X)
        self.valores_imputacion_ = {
            "OverallQual": df["OverallQual"].median(),
            "ExterQual": df["ExterQual"].mode()[0],
            "KitchenQual": df["KitchenQual"].mode()[0],
            "CentralAir": df["CentralAir"].mode()[0],
            "1stFlrSF": df["1stFlrSF"].median(),
            "GrLivArea": df["GrLivArea"].median(),
        }
        df = self._imputar(df)
        self.garagefinish_categorias_ = sorted(df["GarageFinish"].unique())
        self.centralair_categorias_ = sorted(df["CentralAir"].unique())
        self.scaler_ = StandardScaler().fit(df[self.COLS_A_ESCALAR])
        self.columnas_finales_ = self._codificar_y_escalar(df).columns.tolist()
        return self

    def _codificar_y_escalar(self, df):
        df = df.copy()
        for col, categorias in [
            ("GarageFinish", self.garagefinish_categorias_),
            ("CentralAir", self.centralair_categorias_),
        ]:
            dummies = pd.get_dummies(df[col], prefix=col).astype(int)
            dummies = dummies.reindex(columns=[f"{col}_{c}" for c in categorias], fill_value=0)
            df = df.drop(columns=[col]).join(dummies)
        df[self.COLS_A_ESCALAR] = self.scaler_.transform(df[self.COLS_A_ESCALAR])
        return df

    def transform(self, X):
        df = self._imputar(self._preparar(X))
        df = self._codificar_y_escalar(df)[self.columnas_finales_]
        # Garantiza una fila de salida por cada fila de entrada
        assert len(df) == len(X) and not df.isna().any().any()
        return df
