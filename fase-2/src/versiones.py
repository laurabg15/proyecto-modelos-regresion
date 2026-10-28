"""Almacenamiento y carga de versiones del modelo.

Cada version es una carpeta: <output_dir>/<version>/ con
modelo.joblib, metrics.json y metadata.json.
"""
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import joblib

ARCHIVO_MODELO = "modelo.joblib"
ARCHIVO_METRICAS = "metrics.json"
ARCHIVO_METADATOS = "metadata.json"
PATRON_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ErrorVersion(Exception):
    """Error previsible al guardar o cargar una version del modelo."""


def validar_identificador(version):
    if not PATRON_VERSION.match(version):
        raise ErrorVersion(
            f"Identificador de version invalido: '{version}'. "
            "Use letras, numeros, punto, guion o guion bajo (ej.: v1, v2, 2026-10-21)."
        )


def guardar_version(directorio_salida, version, modelo, metricas, metadatos):
    """Guarda una version nueva. Nunca sobrescribe una version existente."""
    validar_identificador(version)
    directorio_salida = Path(directorio_salida)
    destino = directorio_salida / version
    if destino.exists():
        raise ErrorVersion(
            f"La version '{version}' ya existe en {destino}. "
            "Use otro identificador con --version; la version existente no se modifica."
        )
    directorio_salida.mkdir(parents=True, exist_ok=True)

    # Se escribe en una carpeta temporal y se renombra al final: si algo falla,
    # no queda una version incompleta con apariencia de valida
    temporal = Path(tempfile.mkdtemp(prefix=f".{version}-tmp-", dir=directorio_salida))
    try:
        joblib.dump(modelo, temporal / ARCHIVO_MODELO)
        _escribir_json(temporal / ARCHIVO_METRICAS, metricas)
        _escribir_json(temporal / ARCHIVO_METADATOS, metadatos)
        os.rename(temporal, destino)
    except BaseException:
        shutil.rmtree(temporal, ignore_errors=True)
        raise
    return {
        "directorio": destino,
        "modelo": destino / ARCHIVO_MODELO,
        "metricas": destino / ARCHIVO_METRICAS,
        "metadatos": destino / ARCHIVO_METADATOS,
    }


def cargar_modelo(ruta):
    """Carga un modelo desde la carpeta de la version o desde el archivo .joblib."""
    ruta = Path(ruta)
    if ruta.is_dir():
        ruta = ruta / ARCHIVO_MODELO
    if not ruta.is_file():
        raise ErrorVersion(f"No existe el modelo: {ruta}")
    try:
        return joblib.load(ruta)
    except Exception as error:
        raise ErrorVersion(f"No se pudo cargar el modelo {ruta}: {error}") from error


def ultima_version(directorio_modelos):
    """Devuelve la carpeta de la version mas reciente, segun su fecha de creacion."""
    directorio_modelos = Path(directorio_modelos)
    versiones = []
    for carpeta in directorio_modelos.glob("*"):
        metadatos = carpeta / ARCHIVO_METADATOS
        if carpeta.name.startswith(".") or not (carpeta / ARCHIVO_MODELO).is_file() or not metadatos.is_file():
            continue
        with open(metadatos, encoding="utf-8") as f:
            fecha = datetime.fromisoformat(json.load(f)["fecha_creacion"])
        versiones.append((fecha, carpeta))
    if not versiones:
        raise ErrorVersion(f"No hay versiones guardadas en {directorio_modelos}")
    return max(versiones)[1]


def huella_codigo(directorio):
    """Huella SHA-256 del codigo fuente (train.py, predict.py y src/*.py), sin usar git.

    Identifica el codigo que genero una version: cualquier cambio en esos archivos
    produce una huella distinta. Los saltos de linea se normalizan para que la huella
    sea la misma en Windows, Linux y Docker.
    """
    directorio = Path(directorio)
    archivos = sorted(
        p for p in [directorio / "train.py", directorio / "predict.py", *directorio.glob("src/*.py")]
        if p.is_file()
    )
    sha = hashlib.sha256()
    for archivo in archivos:
        sha.update(archivo.relative_to(directorio).as_posix().encode())
        sha.update(archivo.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "sha256": sha.hexdigest(),
        "archivos": [a.relative_to(directorio).as_posix() for a in archivos],
    }


def _escribir_json(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)
        f.write("\n")
