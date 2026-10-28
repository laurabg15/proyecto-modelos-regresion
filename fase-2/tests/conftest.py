"""Datos y modelo de prueba compartidos por las pruebas.

Se usa una muestra pequena y fija de data/train.csv (sin descargas de red) y todo
se guarda en carpetas temporales de pytest, sin tocar models/ ni outputs/.
"""
from pathlib import Path

import pandas as pd
import pytest

from train import entrenar

RAIZ = Path(__file__).resolve().parent.parent
TAMANO_MUESTRA = 150


def entrenar_version(datos_csv, directorio, version):
    """Entrenamiento rapido: muestra pequena y sin validacion cruzada."""
    return entrenar(data=datos_csv, version=version, output_dir=directorio, seed=42, cv_folds=0, exclude_ids=())


@pytest.fixture(scope="session")
def datos_csv(tmp_path_factory):
    """CSV con una muestra fija de viviendas etiquetadas."""
    muestra = pd.read_csv(RAIZ / "data" / "train.csv").sample(TAMANO_MUESTRA, random_state=0)
    ruta = tmp_path_factory.mktemp("datos") / "muestra_train.csv"
    muestra.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def version_entrenada(datos_csv, tmp_path_factory):
    """Una version entrenada una sola vez y reutilizada por las pruebas de prediccion."""
    directorio = tmp_path_factory.mktemp("models")
    resultado = entrenar_version(datos_csv, directorio, "v1")
    resultado["directorio_modelos"] = directorio
    return resultado


@pytest.fixture
def datos_prediccion():
    """Ejemplo de prediccion del repositorio: 8 viviendas, sin SalePrice, con algunos vacios."""
    return pd.read_csv(RAIZ / "data" / "ejemplo_prediccion.csv")
