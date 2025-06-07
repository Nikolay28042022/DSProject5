# Используем официальный образ PyTorch с поддержкой CUDA.
# Вы можете выбрать другую версию CUDA, если это необходимо, но 11.8 широко поддерживается.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY app/requirements_app.txt .

# Устанавливаем зависимости Python.
# --no-cache-dir для уменьшения размера образа.
# --extra-index-url https://download.pytorch.org/whl/cu118 для PyTorch с CUDA
RUN pip install --no-cache-dir -r requirements_app.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    # Дополнительные пакеты, которые могут понадобиться для OpenCV и других утилит
    && apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git

# Копируем остальное содержимое приложения
COPY app/ .

# Копируем директорию shared_data, чтобы YOLO могла найти модель
COPY shared_data/ /shared_data/

# Обязательно для YOLO, чтобы найти модель
ENV YOLO_MODELS_DIR=/shared_data/models

# Порт, который будет слушать Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "main.py"]