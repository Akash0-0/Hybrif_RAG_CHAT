FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GRADIO_ALLOW_FLAGGING=never
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]
