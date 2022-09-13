FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y \
  libgl1-mesa-dev \
  python3-pip \
  && apt-get clean

RUN pip3 install --no-cache-dir \
  atari_py \
  chunkedfile>=0.3.0 \
  crafter>=1.5.1 \
  gym \
  opencv-python \
  plotly \
  tqdm

COPY . /app
WORKDIR /app
CMD ["python3", "main.py"]
