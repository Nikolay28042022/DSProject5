# Используем официальный образ PyTorch с поддержкой CUDA.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости Python.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    && apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git

# Копируем остальное содержимое приложения (включая main.py, detector.py, templates/)
COPY . .

# Обязательно для YOLO, чтобы найти модель
# Эта переменная окружения все еще нужна
ENV YOLO_MODELS_DIR=/shared_data/models    

# Порт, который будет слушать Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "main.py"]