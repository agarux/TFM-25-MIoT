FROM python:3.9

WORKDIR /edge

# Copiar el contenido al contenedor
COPY . /edge

# Instalamos dependencias necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

ENV NAME=anagarridoupm

# Ejecutar el script cuando el contenedor se inicie
CMD ["python3", "cameras_sim.py"]
