# docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "src.api.app:app", "--reload", "--host", "0.0.0.0", "--port", "5000"]
