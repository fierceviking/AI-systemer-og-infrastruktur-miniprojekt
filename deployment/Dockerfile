FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn onnxruntime numpy

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]