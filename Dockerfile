FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads temp_processing
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
