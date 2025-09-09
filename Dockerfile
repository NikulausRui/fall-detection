FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN apt-get update && apt-get install curl libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY requirements.txt /workspace
WORKDIR /workspace
RUN uv pip install -r requirements.txt --system
COPY . /workspace
ENTRYPOINT ["python", "fall_detetctoin.py"]
