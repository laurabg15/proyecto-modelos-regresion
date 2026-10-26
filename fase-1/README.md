# Predicción del Precio de Viviendas – House Prices (Ames, Iowa)

## Integrantes del equipo
- Laura Vanessa Botero Gil
- Duban Lopez Sanchez

## Descripción del problema
Se busca estimar el precio de venta de viviendas residenciales en Ames, Iowa (EE. UU.) a partir de sus características físicas, de calidad y de ubicación. Es un problema de **aprendizaje supervisado de regresión**, cuya variable objetivo es `SalePrice` (precio de venta en USD).

## Fuente del conjunto de datos
Competencia de Kaggle *House Prices – Advanced Regression Techniques*:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

- `train.csv`: 1.460 viviendas, 79 variables explicativas + `SalePrice`.
- `test.csv`: 1.459 viviendas sin `SalePrice` (conjunto de validación de la competencia).

## Objetivo del modelo
Predecir `SalePrice` con el menor error posible en viviendas no vistas, usando un modelo simple e interpretable.

## Algoritmo utilizado
**Regresión Lasso** (`Lasso(alpha=100)` de scikit-learn), dentro de un `Pipeline` con el transformador de datos `TransformadorCasas`.

Se compararon un modelo base (mediana), Regresión Lineal, Ridge, Lasso, Random Forest y XGBoost mediante validación cruzada de 5 folds dentro del conjunto de entrenamiento. Random Forest y XGBoost mostraron sobreajuste: un error en entrenamiento muy inferior al de validación (R² de 0,98 y 1,00 en entrenamiento frente a 0,81 y 0,79 en validación). Los modelos lineales fueron estables (R² 0,81 en entrenamiento y 0,79 en validación) y prácticamente equivalentes entre sí.

Se eligió Lasso por su estabilidad entre entrenamiento y validación, su simplicidad, su interpretabilidad y porque elimina las variables one-hot redundantes. **Intercambio aceptado:** Random Forest obtuvo un MAE de validación unos USD 3.200 menor; se priorizaron estabilidad e interpretabilidad sobre esa diferencia.

Variables de entrada (10): `OverallQual`, `ExterQual`, `BsmtQual`, `KitchenQual`, `TotalBsmtSF`, `1stFlrSF`, `GrLivArea`, `GarageArea`, `GarageFinish` (one-hot) y `CentralAir` (one-hot).

Tratamiento de faltantes: en Ames, un faltante en sótano o garaje significa que la vivienda no lo tiene (`BsmtQual` = 0 sin sótano, áreas = 0, `GarageFinish` = `SinGaraje`); el resto se imputa con la mediana o la moda del entrenamiento. El modelo devuelve una predicción por cada vivienda de entrada.

## Métrica empleada
- **MAE** (error absoluto medio): métrica principal, en dólares.
- **RMSE** (raíz del error cuadrático medio): penaliza más los errores grandes.
- **R²** (coeficiente de determinación): proporción de la variabilidad del precio que explica el modelo.

## Principales resultados obtenidos
Partición 80/20 del `train.csv` (`random_state=80`). Los modelos se compararon con validación cruzada de 5 folds sobre el 80 % de entrenamiento (1.167 viviendas, sin el atípico Id 1299):

| Modelo                  | MAE Train | MAE Val (CV) | RMSE Val (CV) | R² Train | R² Val (CV) |
|-------------------------|-----------|--------------|---------------|----------|-------------|
| Modelo Base (Mediana)   | 55.950    | 55.982       | 82.726        | -0,05    | -0,05       |
| Regresión Lineal        | 23.535    | 23.946       | 36.316        | 0,81     | 0,79        |
| Ridge (alpha=10)        | 23.489    | 23.872       | 36.294        | 0,81     | 0,79        |
| **Lasso (alpha=100)**   | **23.518**| **23.920**   | **36.313**    | **0,81** | **0,79**    |
| Random Forest           | 7.737     | 20.742       | 33.824        | 0,98     | 0,81        |
| XGBoost                 | 2.271     | 21.959       | 35.781        | 1,00     | 0,79        |

**Evaluación final del modelo entregado** (`modelo_casas.joblib`, entrenado con las 1.167 viviendas y evaluado una sola vez en el 20 % de prueba, 292 viviendas):

| Métrica | Prueba     |
|---------|------------|
| MAE     | USD 21.003 |
| RMSE    | USD 27.679 |
| R²      | 0,855      |

- El modelo se equivoca en promedio unos **USD 21.000** por vivienda en prueba, frente a unos USD 56.000 del modelo base.
- Explica alrededor del **86 %** de la variabilidad del precio en el conjunto de prueba (≈ 79 % en validación cruzada).

## Instrucciones para ejecutar el notebook
1. Clonar el repositorio y entrar a la carpeta `fase-1`.
2. Crear un entorno virtual (opcional) e instalar las dependencias con las versiones usadas (Python 3.12):
   ```bash
   pip install -r requirements.txt jupyter
   ```
3. Datos:
   - Si `train.csv` y `test.csv` ya están en la carpeta, el notebook omite la descarga automáticamente.
   - Si no están, el notebook los descarga desde Kaggle. Para eso, genere un token en https://www.kaggle.com/settings/api y guárdelo en un archivo `.env` en la carpeta `fase-1`:
     ```
     KAGGLE_API_TOKEN=su_token
     ```
     (Debe tener aceptadas las reglas de la competencia en Kaggle.)
4. Abrir el notebook y ejecutar todas las celdas en orden:
   ```bash
   jupyter notebook InitialExploration_Dataset.ipynb
   ```

## Archivos del proyecto
- `InitialExploration_Dataset.ipynb`: exploración, preparación, comparación de modelos y evaluación.
- `transformador_casas.py`: clase `TransformadorCasas` con todas las transformaciones de datos.
- `modelo_casas.joblib`: pipeline final (transformador + modelo), el mismo evaluado en el conjunto de prueba.
- `requirements.txt`: versiones exactas de las dependencias.

### Uso del modelo guardado
Ejecutar desde la carpeta `fase-1` (donde está `transformador_casas.py`):
```python
import joblib, pandas as pd

modelo = joblib.load("modelo_casas.joblib")

datos = pd.read_csv("test.csv")
predicciones = modelo.predict(datos)  # una predicción por cada vivienda
```
