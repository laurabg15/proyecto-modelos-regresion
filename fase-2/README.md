# Fase 2 – Predicción del Precio de Viviendas

**Integrantes:** Laura Vanessa Botero Gil · Duban Lopez Sanchez

**Versión entregada:** etiqueta de Git `entrega-2`

## Problema y relación con la Fase 1
Estimar `SalePrice` (USD) de viviendas de Ames, Iowa. La [Fase 1](../fase-1) seleccionó el modelo en un notebook; esta fase lo convierte en scripts de entrenamiento y predicción, con versionamiento, pruebas y Docker. Modelo: **Lasso (alpha = 100)** dentro de un `Pipeline` con el transformador `TransformadorCasas`.

## Correcciones respecto a la Fase 1
- Modelos y alpha comparados con **validación cruzada (5 folds) dentro del entrenamiento**; la prueba se usa una sola vez.
- `BsmtQual` vacío = **sin sótano** (0), no la media. Vacíos de garaje = sin garaje.
- Ninguna fila se descarta: **una predicción por fila de entrada**.
- `TransformadorCasas` en un módulo importable y un único `Pipeline` guardado; las métricas corresponden al artefacto guardado (sin el atípico Id 1299).
- Elección de Lasso con el intercambio explícito: Random Forest tuvo menor MAE en validación cruzada.
- `requirements.txt` con versiones; PR revisados por el otro integrante.

## Instalación
Python **3.12.2**. Comandos para **PowerShell**, desde `fase-2`:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos
- **Fuente:** [Kaggle – House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). `data/train.csv` (1.460 viviendas) ya está en el repositorio.
- **`data/ejemplo_prediccion.csv`:** 8 viviendas sin `SalePrice`, con el esquema real; tres tienen vacíos.

**Esquema de predicción:** el CSV debe tener estas 10 columnas, en cualquier orden.

| Columna | Tipo | Si está vacía |
|---|---|---|
| `OverallQual`, `1stFlrSF`, `GrLivArea` | numérica | mediana del entrenamiento |
| `TotalBsmtSF`, `GarageArea` | numérica | 0 (no tiene) |
| `ExterQual`, `KitchenQual` (Po–Ex), `CentralAir` (Y/N) | categórica | moda del entrenamiento |
| `BsmtQual` (Po–Ex) | categórica | 0 (sin sótano) |
| `GarageFinish` (Fin/RFn/Unf) | categórica | `SinGaraje` |

`Id` es opcional y se copia a la salida. Las columnas adicionales se ignoran. Una calidad fuera de Po–Ex se trata como vacía; un valor desconocido de `CentralAir` o `GarageFinish` deja sus columnas one-hot en 0. Una columna ausente o texto en una columna numérica producen un error.

## Uso
```powershell
python train.py --help
python predict.py --help

python train.py --data data/train.csv --target SalePrice --output-dir models --version v3 --seed 80
python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```
- `train.py` separa 80/20 **antes** de ajustar transformaciones, evalúa el modelo y el modelo base (mediana) en prueba y rechaza una versión existente sin modificarla.
- `predict.py` no reentrena. `--model` acepta la carpeta o el `.joblib`; **si se omite, usa la versión más reciente**. La salida tiene `Id` y `prediccion`, una fila por entrada y en el mismo orden. Ante un error muestra un mensaje, sale con código 1 y no genera el archivo.

## Versiones y resultados
- `models/<version>/`: `modelo.joblib` (pipeline completo), `metrics.json` (modelo y modelo base) y `metadata.json` (datos, partición, parámetros, semilla, entorno y huella SHA-256 del código).
- `outputs/`: predicciones generadas (excluida de Git).

| Versión | Semilla | MAE modelo | MAE base | R² | Generada |
|---|---|---|---|---|---|
| `v1` | 80 | 21.002,85 | 54.044,78 | 0,855 | Local (código del commit `569fbcf`) |
| `v2` | 42 | 23.906,90 | 59.568,25 | 0,821 | Local (código final) |
| `v-docker` | 80 | 21.002,85 | 54.044,78 | 0,855 | Docker |

El modelo reduce el MAE del modelo base en ~60 %. Métrica principal: MAE en prueba (292 viviendas).

**Limitaciones:** la métrica depende de la partición (ver `v1` frente a `v2`); Random Forest obtuvo ≈ USD 3.200 menos de MAE en validación cruzada; las variables se seleccionaron una vez en el EDA; el error crece en las viviendas más caras; los datos son de Ames entre 2006 y 2010.

## Docker
Con Docker Desktop abierto, desde `fase-2` (PowerShell; en Bash usar `$PWD` y `\`):
```powershell
docker build -t modelos-fase2:1.0 .

docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" `
  modelos-fase2:1.0 python train.py --data data/train.csv --target SalePrice --output-dir models --version v-docker2 --seed 80

docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" `
  modelos-fase2:1.0 python predict.py --model models/v-docker --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```
La imagen (`python:3.12.2-slim`) contiene solo el código y no entrena durante el `build`. Datos, modelos y predicciones se montan desde el equipo, así que **persisten al eliminar el contenedor**.

## Pruebas
```powershell
python -m pytest -v
```
13 pruebas con una muestra de 150 viviendas y carpetas temporales. Cubren entrenamiento y almacenamiento, carga y predicción (resultados iguales antes y después de guardar), valores faltantes, entrada inválida (error claro y código ≠ 0) y versionamiento (una versión nueva no modifica la anterior).

## Evidencia de ejecución
```
> python train.py --data data/train.csv --target SalePrice --output-dir models --version v2 --seed 42
  Metrica (prueba)          Modelo   Modelo base
  MAE                    23,906.90     59,568.25
  RMSE                   37,097.03     88,667.17
  R2                          0.82         -0.02
  Archivos generados: models/v2/modelo.joblib, metrics.json, metadata.json

> python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
Predicciones generadas: 8 (una por fila de data/ejemplo_prediccion.csv)

> python -m pytest
13 passed in 12.74s
```
En Docker, `v-docker` obtuvo las mismas métricas que `v1` y permaneció en `models/` tras eliminar el contenedor. Una columna ausente produce `Error: Faltan columnas requeridas: ['GrLivArea']` con código de salida 1.

## Fuentes
- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data*. Journal of Statistics Education, 19(3).
- Documentación de scikit-learn, pandas, pytest y Docker.
- Se utilizó Claude para la asistencia en redacción del codigo,  el código fue revisado y probado por el equipo.
