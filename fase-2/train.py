"""Entrena una version del modelo de precios de viviendas y la guarda en disco.

Ejemplo (desde la carpeta fase-2):
    python train.py --data data/train.csv --target SalePrice --output-dir models --version v1 --seed 80
"""
import argparse
import platform
import sys
from datetime import datetime
from importlib.metadata import version as version_paquete
from pathlib import Path

from src.datos import ErrorDatos, leer_csv, validar_columnas, validar_objetivo, validar_tipos
from src.modelo import (
    calcular_metricas, construir_modelo_base, construir_pipeline,
    excluir_por_id, separar_datos, validacion_cruzada,
)
from src.preprocesamiento import ESQUEMA_ENTRADA
from src.versiones import ErrorVersion, guardar_version, hash_archivo, info_codigo, validar_identificador

FUENTE_DATOS = "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data"
DIRECTORIO_CODIGO = Path(__file__).resolve().parent


def construir_parser():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo Lasso de precios de viviendas (Ames) y guarda una version nueva.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV de datos etiquetados (ej.: data/train.csv)")
    parser.add_argument("--target", default="SalePrice", help="Nombre de la variable objetivo")
    parser.add_argument("--output-dir", default="models", help="Carpeta donde se crean las versiones")
    parser.add_argument("--version", required=True, help="Identificador unico de la version (ej.: v1)")
    parser.add_argument("--seed", type=int, default=80, help="Semilla de la particion y la validacion cruzada")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcion de datos para prueba")
    parser.add_argument("--alpha", type=float, default=100.0, help="Regularizacion de Lasso")
    parser.add_argument("--max-iter", type=int, default=10000, help="Iteraciones maximas de Lasso")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Folds de validacion cruzada sobre entrenamiento (0 para omitirla)")
    parser.add_argument("--id-column", default="Id", help="Columna identificadora de cada vivienda")
    parser.add_argument("--exclude-ids", type=int, nargs="*", default=[1299],
                        help="Ids de atipicos que se excluyen solo del entrenamiento")
    return parser


def entrenar(data, version, target="SalePrice", output_dir="models", seed=80, test_size=0.2,
             alpha=100.0, max_iter=10000, cv_folds=5, id_column="Id", exclude_ids=(1299,)):
    """Ejecuta el entrenamiento completo y devuelve las rutas, metricas y metadatos generados."""
    validar_identificador(version)
    if (Path(output_dir) / version).exists():
        raise ErrorVersion(
            f"La version '{version}' ya existe en {Path(output_dir) / version}. "
            "Use otro identificador con --version; la version existente no se modifica."
        )
    if not 0 < test_size < 1:
        raise ErrorDatos("--test-size debe estar entre 0 y 1")

    # 1. Lectura y validacion
    df = leer_csv(data)
    validar_objetivo(df, target)
    validar_columnas(df)
    df = validar_tipos(df)

    # 2. Separacion ANTES de ajustar cualquier transformacion
    X_train, X_test, y_train, y_test = separar_datos(df, target, test_size, seed)
    X_train, y_train, ids_excluidos = excluir_por_id(X_train, y_train, id_column, list(exclude_ids))

    # 3. Validacion cruzada solo sobre entrenamiento (el pipeline se reajusta en cada fold)
    cv = None
    if cv_folds >= 2:
        cv = {
            "folds": cv_folds,
            "modelo": validacion_cruzada(construir_pipeline(alpha, max_iter), X_train, y_train, cv_folds, seed),
            "modelo_base": validacion_cruzada(construir_modelo_base(), X_train, y_train, cv_folds, seed),
        }

    # 4. Entrenamiento con todo el entrenamiento y evaluacion unica en prueba
    modelo = construir_pipeline(alpha, max_iter).fit(X_train, y_train)
    modelo_base = construir_modelo_base().fit(X_train, y_train)

    metricas = {
        "conjunto_evaluacion": f"prueba ({len(X_test)} observaciones no usadas para ajustar)",
        "metrica_principal": "MAE",
        "modelo": calcular_metricas(y_test, modelo.predict(X_test)),
        "modelo_base": calcular_metricas(y_test, modelo_base.predict(X_test)),
        "modelo_en_entrenamiento": calcular_metricas(y_train, modelo.predict(X_train)),
        "validacion_cruzada_entrenamiento": cv,
    }
    mae_modelo, mae_base = metricas["modelo"]["MAE"], metricas["modelo_base"]["MAE"]
    metricas["reduccion_MAE_vs_base_pct"] = round(100 * (mae_base - mae_modelo) / mae_base, 2)

    metadatos = {
        "version": version,
        "fecha_creacion": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tipo_problema": "regresion",
        "variable_objetivo": target,
        "algoritmo": "Lasso (scikit-learn) dentro de un Pipeline con TransformadorCasas",
        "parametros_modelo": {"alpha": alpha, "max_iter": max_iter},
        "semilla": seed,
        "variables_entrada": ESQUEMA_ENTRADA,
        "variables_tras_transformacion": modelo.named_steps["transformador"].columnas_finales_,
        "tratamiento_faltantes": (
            "Sotano/garaje ausente: BsmtQual=0, TotalBsmtSF=0, GarageArea=0, GarageFinish='SinGaraje'. "
            "Resto: mediana (numericas) o moda (categoricas) aprendidas en entrenamiento."
        ),
        "datos": {
            "fuente": FUENTE_DATOS,
            "archivo": Path(data).as_posix(),
            "sha256": hash_archivo(data),
            "observaciones": len(df),
            "columnas": df.shape[1],
        },
        "separacion": {
            "estrategia": "train_test_split aleatorio, antes de ajustar transformaciones",
            "test_size": test_size,
            "observaciones_entrenamiento": len(X_train),
            "observaciones_prueba": len(X_test),
            "ids_excluidos_de_entrenamiento": ids_excluidos,
        },
        "metrica_evaluacion": "MAE (principal), RMSE y R2 sobre el conjunto de prueba",
        "entorno": {
            "python": platform.python_version(),
            "plataforma": platform.platform(),
            "dependencias": "requirements.txt",
            "librerias": {p: version_paquete(p) for p in ["pandas", "numpy", "scikit-learn", "joblib"]},
        },
        "codigo": info_codigo(DIRECTORIO_CODIGO),
    }

    rutas = guardar_version(output_dir, version, modelo, metricas, metadatos)
    return {"rutas": rutas, "metricas": metricas, "metadatos": metadatos}


def imprimir_resumen(resultado):
    m = resultado["metricas"]
    meta = resultado["metadatos"]
    print(f"\nVersion '{meta['version']}' entrenada correctamente")
    print(f"  Entrenamiento: {meta['separacion']['observaciones_entrenamiento']} | "
          f"Prueba: {meta['separacion']['observaciones_prueba']} | Semilla: {meta['semilla']}")
    print(f"\n  {'Metrica (prueba)':18s}{'Modelo':>14s}{'Modelo base':>14s}")
    for nombre in ["MAE", "RMSE", "R2"]:
        print(f"  {nombre:18s}{m['modelo'][nombre]:>14,.2f}{m['modelo_base'][nombre]:>14,.2f}")
    print(f"\n  Reduccion del MAE frente al modelo base: {m['reduccion_MAE_vs_base_pct']} %")
    if m["validacion_cruzada_entrenamiento"]:
        cv = m["validacion_cruzada_entrenamiento"]["modelo"]["MAE"]
        print(f"  MAE en validacion cruzada: {cv['media']:,.2f} (+/- {cv['desviacion']:,.2f})")
    print("\n  Archivos generados:")
    for clave in ["modelo", "metricas", "metadatos"]:
        print(f"    {resultado['rutas'][clave].as_posix()}")


def main(argv=None):
    args = construir_parser().parse_args(argv)
    try:
        resultado = entrenar(
            data=args.data, version=args.version, target=args.target, output_dir=args.output_dir,
            seed=args.seed, test_size=args.test_size, alpha=args.alpha, max_iter=args.max_iter,
            cv_folds=args.cv_folds, id_column=args.id_column, exclude_ids=args.exclude_ids,
        )
    except (ErrorDatos, ErrorVersion) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    imprimir_resumen(resultado)
    return 0


if __name__ == "__main__":
    sys.exit(main())
