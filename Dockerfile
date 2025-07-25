FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

ENTRYPOINT ["bash"]
