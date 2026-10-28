"""Pruebas de la aplicacion de la Fase 2. Ejecutar desde fase-2 con: python -m pytest"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import RAIZ, TAMANO_MUESTRA, entrenar_version
from predict import COLUMNA_PREDICCION, predecir
from src.datos import ErrorDatos
from src.preprocesamiento import ESCALA_CALIDAD, ESQUEMA_ENTRADA
from src.versiones import ErrorVersion, cargar_modelo, ultima_version


def ejecutar_script(*argumentos):
    """Ejecuta un script como lo haria un usuario en la terminal."""
    return subprocess.run([sys.executable, *argumentos], cwd=RAIZ, capture_output=True, text=True)


def contenido_version(directorio):
    return {archivo.name: archivo.read_bytes() for archivo in sorted(directorio.iterdir())}


# --- 1. Entrenamiento y almacenamiento ---

def test_entrenamiento_genera_modelo_metricas_y_metadatos(version_entrenada):
    rutas = version_entrenada["rutas"]
    for clave in ["modelo", "metricas", "metadatos"]:
        assert rutas[clave].is_file()

    assert hasattr(cargar_modelo(rutas["directorio"]), "predict")

    metricas = json.loads(rutas["metricas"].read_text(encoding="utf-8"))
    for modelo in ["modelo", "modelo_base"]:
        assert set(metricas[modelo]) == {"MAE", "RMSE", "R2"}
    assert metricas["modelo"]["MAE"] < metricas["modelo_base"]["MAE"]

    metadatos = json.loads(rutas["metadatos"].read_text(encoding="utf-8"))
    assert metadatos["version"] == "v1"
    assert metadatos["semilla"] == 42
    assert metadatos["variable_objetivo"] == "SalePrice"
    assert metadatos["variables_entrada"] == ESQUEMA_ENTRADA
    assert metadatos["datos"]["observaciones"] == TAMANO_MUESTRA
    separacion = metadatos["separacion"]
    assert separacion["observaciones_entrenamiento"] + separacion["observaciones_prueba"] == TAMANO_MUESTRA


# --- 2. Carga y prediccion ---

def test_modelo_cargado_predice_una_fila_por_entrada_y_conserva_resultados(version_entrenada, datos_prediccion):
    modelo_cargado = cargar_modelo(version_entrenada["rutas"]["directorio"])

    antes_de_guardar = version_entrenada["modelo"].predict(datos_prediccion)
    despues_de_cargar = modelo_cargado.predict(datos_prediccion)

    assert len(despues_de_cargar) == len(datos_prediccion)
    np.testing.assert_allclose(despues_de_cargar, antes_de_guardar, rtol=1e-9)


def test_prediccion_conserva_orden_e_id(version_entrenada, datos_prediccion):
    salida = predecir(version_entrenada["modelo"], datos_prediccion)

    assert list(salida.columns) == ["Id", COLUMNA_PREDICCION]
    assert salida["Id"].tolist() == datos_prediccion["Id"].tolist()
    assert salida[COLUMNA_PREDICCION].notna().all()


def test_orden_de_columnas_y_columnas_extra_no_cambian_la_prediccion(version_entrenada, datos_prediccion):
    modelo = version_entrenada["modelo"]
    reordenado = datos_prediccion[datos_prediccion.columns[::-1]].assign(ColumnaExtra="x", SalePrice=1)

    np.testing.assert_allclose(
        predecir(modelo, reordenado)[COLUMNA_PREDICCION],
        predecir(modelo, datos_prediccion)[COLUMNA_PREDICCION],
    )


# --- 3. Valores faltantes ---

def test_nulos_se_procesan_segun_la_estrategia_documentada(version_entrenada, datos_prediccion):
    modelo = version_entrenada["modelo"]
    base = datos_prediccion[datos_prediccion["Id"] == 1461]

    def variante(**cambios):
        fila = base.copy()
        for columna, valor in cambios.items():
            fila[columna] = pd.Series([valor], index=fila.index, dtype=object)
        return fila

    # Sotano ausente: un vacio equivale a "no tiene sotano" (area 0)
    sin_sotano_nulo = variante(BsmtQual=np.nan, TotalBsmtSF=np.nan)
    sin_sotano_cero = variante(BsmtQual=np.nan, TotalBsmtSF=0)
    # Garaje ausente: un vacio equivale a "no tiene garaje" (area 0)
    sin_garaje_nulo = variante(GarageFinish=np.nan, GarageArea=np.nan)
    sin_garaje_cero = variante(GarageFinish=np.nan, GarageArea=0)
    # Otros vacios: se imputan con la moda aprendida en entrenamiento
    moda = version_entrenada["modelo"].named_steps["transformador"].valores_imputacion_["KitchenQual"]
    cocina_nula = variante(KitchenQual=np.nan)
    cocina_moda = variante(KitchenQual={v: k for k, v in ESCALA_CALIDAD.items()}[moda])

    entradas = pd.concat([sin_sotano_nulo, sin_sotano_cero, sin_garaje_nulo, sin_garaje_cero, cocina_nula, cocina_moda])
    salida = predecir(modelo, entradas)[COLUMNA_PREDICCION].to_numpy()

    assert len(salida) == len(entradas)
    assert np.isfinite(salida).all()
    assert salida[0] == pytest.approx(salida[1])
    assert salida[2] == pytest.approx(salida[3])
    assert salida[4] == pytest.approx(salida[5])


# --- 4. Entrada invalida ---

def test_columna_requerida_ausente_produce_error_claro(version_entrenada, datos_prediccion):
    with pytest.raises(ErrorDatos, match="GrLivArea"):
        predecir(version_entrenada["modelo"], datos_prediccion.drop(columns=["GrLivArea"]))


def test_texto_en_columna_numerica_produce_error_claro(version_entrenada, datos_prediccion):
    datos = datos_prediccion.astype({"GarageArea": object})
    datos.loc[0, "GarageArea"] = "grande"
    with pytest.raises(ErrorDatos, match="GarageArea"):
        predecir(version_entrenada["modelo"], datos)


def test_cli_con_entrada_invalida_sale_con_error_y_sin_archivo(version_entrenada, datos_prediccion, tmp_path):
    entrada = tmp_path / "sin_columna.csv"
    datos_prediccion.drop(columns=["GrLivArea"]).to_csv(entrada, index=False)
    salida = tmp_path / "predicciones.csv"

    resultado = ejecutar_script(
        "predict.py", "--model", str(version_entrenada["rutas"]["directorio"]),
        "--input", str(entrada), "--output", str(salida),
    )

    assert resultado.returncode != 0
    assert "Faltan columnas requeridas" in resultado.stderr
    assert not salida.exists()


def test_cli_con_modelo_inexistente_sale_con_error(tmp_path):
    resultado = ejecutar_script(
        "predict.py", "--model", str(tmp_path / "v9"),
        "--input", "data/ejemplo_prediccion.csv", "--output", str(tmp_path / "salida.csv"),
    )

    assert resultado.returncode != 0
    assert "No existe el modelo" in resultado.stderr


# --- 5. Versionamiento ---

def test_segunda_version_conserva_la_primera_y_no_se_sobrescribe(datos_csv, datos_prediccion, tmp_path):
    v1 = entrenar_version(datos_csv, tmp_path, "v1")["rutas"]["directorio"]
    contenido_v1 = contenido_version(v1)

    v2 = entrenar_version(datos_csv, tmp_path, "v2")["rutas"]["directorio"]

    assert contenido_version(v1) == contenido_v1
    for version in [v1, v2]:
        assert len(cargar_modelo(version).predict(datos_prediccion)) == len(datos_prediccion)
    assert ultima_version(tmp_path) == v2

    with pytest.raises(ErrorVersion, match="ya existe"):
        entrenar_version(datos_csv, tmp_path, "v1")
    assert contenido_version(v1) == contenido_v1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v1", "v2"]


def test_cli_predict_sin_model_usa_la_version_mas_reciente(datos_csv, tmp_path):
    entrenar_version(datos_csv, tmp_path, "v1")
    entrenar_version(datos_csv, tmp_path, "v2")

    resultado = ejecutar_script(
        "predict.py", "--models-dir", str(tmp_path),
        "--input", "data/ejemplo_prediccion.csv", "--output", str(tmp_path / "salida.csv"),
    )

    assert resultado.returncode == 0
    assert "v2" in resultado.stdout
    assert len(pd.read_csv(tmp_path / "salida.csv")) == 8


# --- Ayuda de los scripts ---

@pytest.mark.parametrize("script", ["train.py", "predict.py"])
def test_scripts_muestran_ayuda(script):
    resultado = ejecutar_script(script, "--help")

    assert resultado.returncode == 0
    assert "--help" in resultado.stdout
