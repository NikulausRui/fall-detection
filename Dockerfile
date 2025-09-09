FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
COPY . /workspace
WORKDIR /workspace
RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "fall_detection.py"]