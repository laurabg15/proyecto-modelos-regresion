"""Almacenamiento y carga de versiones del modelo.

Cada version es una carpeta: <output_dir>/<version>/ con
modelo.joblib, metrics.json y metadata.json.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
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


def hash_archivo(ruta):
    sha = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            sha.update(bloque)
    return sha.hexdigest()


def info_codigo(directorio):
    """Commit de git del codigo. En Docker (sin .git) se toma de la variable GIT_COMMIT."""
    if os.environ.get("GIT_COMMIT"):
        return {"commit": os.environ["GIT_COMMIT"], "cambios_sin_commit": None, "fuente": "variable GIT_COMMIT"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directorio, capture_output=True, text=True, check=True
        ).stdout.strip()
        cambios = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=directorio, capture_output=True, text=True, check=True
        ).stdout.strip()
        return {"commit": commit, "cambios_sin_commit": bool(cambios), "fuente": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "desconocido", "cambios_sin_commit": None, "fuente": "no disponible"}


def _escribir_json(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)
        f.write("\n")
