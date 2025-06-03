FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /server

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /server

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Instalar mmengine y mmcv
RUN pip install -U openmim \
    && mim install mmengine==0.10.5 \
    && mim install mmcv==2.1.0

# Instalar mmsegmentation
RUN pip install mmsegmentation==1.2.2

ENV TORCH_CXX11_ABI=0

CMD ["python3", "main_mqtt_handler_cpu.py"]