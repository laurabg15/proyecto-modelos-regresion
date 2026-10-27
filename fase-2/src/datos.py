"""Lectura y validacion de los datos de entrada."""
from pathlib import Path

import pandas as pd

from src.preprocesamiento import ESQUEMA_ENTRADA


class ErrorDatos(Exception):
    """Error en los datos de entrada."""


def leer_csv(ruta):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorDatos(f"No existe el archivo de datos: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ErrorDatos(f"No se pudo leer {ruta} como CSV: {error}") from error
    if df.empty:
        raise ErrorDatos(f"El archivo {ruta} no tiene filas")
    return df


def validar_columnas(df, columnas=ESQUEMA_ENTRADA):
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ErrorDatos(f"Faltan columnas requeridas: {faltantes}")


def validar_tipos(df, esquema=ESQUEMA_ENTRADA):
    """Convierte las columnas numericas y rechaza valores que no sean numeros.

    Una celda vacia (NaN) es valida: la imputacion del modelo la trata.
    """
    df = df.copy()
    for columna, tipo in esquema.items():
        if tipo != "numerica":
            continue
        convertida = pd.to_numeric(df[columna], errors="coerce")
        invalidas = convertida.isna() & df[columna].notna()
        if invalidas.any():
            filas = invalidas[invalidas].index[:5].tolist()
            ejemplos = df.loc[filas, columna].tolist()
            raise ErrorDatos(
                f"La columna '{columna}' debe ser numerica; "
                f"valores invalidos en las filas {filas}: {ejemplos}"
            )
        df[columna] = convertida
    return df


def validar_objetivo(df, objetivo):
    if objetivo not in df.columns:
        raise ErrorDatos(f"No existe la variable objetivo '{objetivo}' en los datos")
    valores = pd.to_numeric(df[objetivo], errors="coerce")
    if valores.isna().any():
        raise ErrorDatos(
            f"La variable objetivo '{objetivo}' tiene {int(valores.isna().sum())} "
            "valores vacios o no numericos"
        )
