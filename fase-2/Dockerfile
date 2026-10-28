# Imagen base: Python 3.12.2 (la misma version usada en local), variante liviana
FROM python:3.12.2-slim

# Carpeta de trabajo dentro del contenedor; los comandos se ejecutan desde aqui
WORKDIR /app

# No generar archivos .pyc y mostrar los mensajes de Python sin retraso
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Primero solo las dependencias: si el codigo cambia pero requirements.txt no,
# Docker reutiliza esta capa y no vuelve a instalar todo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la aplicacion. Datos, modelos y resultados NO se copian:
# se montan desde el equipo anfitrion al ejecutar el contenedor
COPY src/ src/
COPY train.py predict.py ./

# Sin ENTRYPOINT: se elige el script al ejecutar (python train.py ... o python predict.py ...).
# Si no se indica ningun comando, se muestra la ayuda del entrenamiento
CMD ["python", "train.py", "--help"]
