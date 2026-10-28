"""Genera predicciones de precio con una version guardada del modelo.

No reentrena ni reajusta transformaciones: aplica el Pipeline guardado.

Ejemplo (desde la carpeta fase-2):
    python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.datos import ErrorDatos, leer_csv, validar_columnas, validar_tipos
from src.versiones import ErrorVersion, cargar_modelo, ultima_version

COLUMNA_PREDICCION = "prediccion"


def construir_parser():
    parser = argparse.ArgumentParser(
        description="Predice SalePrice (USD) con una version guardada del modelo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None,
                        help="Carpeta de la version (ej.: models/v1) o archivo modelo.joblib. "
                             "Si se omite, se usa la version mas reciente de --models-dir")
    parser.add_argument("--models-dir", default="models",
                        help="Carpeta de versiones donde se busca la mas reciente si no se indica --model")
    parser.add_argument("--input", required=True, help="CSV con las variables de entrada (sin SalePrice)")
    parser.add_argument("--output", required=True, help="CSV de salida con las predicciones")
    parser.add_argument("--id-column", default="Id",
                        help="Columna identificadora que se copia a la salida si existe en la entrada")
    return parser


def predecir(modelo, df, id_column="Id"):
    """Devuelve un DataFrame con una prediccion por fila, en el mismo orden de la entrada.

    Las columnas adicionales se ignoran; el Pipeline selecciona y ordena las que necesita.
    """
    validar_columnas(df)
    df = validar_tipos(df)
    predicciones = modelo.predict(df)
    if len(predicciones) != len(df):
        raise ErrorDatos(f"Se esperaban {len(df)} predicciones y se obtuvieron {len(predicciones)}")
    salida = pd.DataFrame({COLUMNA_PREDICCION: predicciones})
    if id_column in df.columns:
        salida.insert(0, id_column, df[id_column].to_numpy())
    return salida


def guardar_csv(df, ruta):
    """Escribe en un archivo temporal y lo renombra al final, para no dejar salidas a medias."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(prefix=f".{ruta.name}-", suffix=".tmp", dir=ruta.parent)
    os.close(descriptor)
    try:
        df.to_csv(temporal, index=False)
        os.replace(temporal, ruta)
    except BaseException:
        Path(temporal).unlink(missing_ok=True)
        raise


def main(argv=None):
    args = construir_parser().parse_args(argv)
    try:
        ruta_modelo = Path(args.model) if args.model else ultima_version(args.models_dir)
        modelo = cargar_modelo(ruta_modelo)
        datos = leer_csv(args.input)
        resultado = predecir(modelo, datos, args.id_column)
        guardar_csv(resultado, args.output)
    except (ErrorDatos, ErrorVersion) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Modelo utilizado: {ruta_modelo.as_posix()}")
    print(f"Predicciones generadas: {len(resultado)} (una por fila de {args.input})")
    print(f"Archivo de salida: {Path(args.output).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
