# Используем официальный образ PyTorch с поддержкой CUDA.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt отдельно, чтобы изменения в коде приложения
# не приводили к переустановке зависимостей при каждом билде.
COPY requirements.txt .

# Устанавливаем зависимости Python и системные пакеты.
# Убираем --no-cache-dir для ускорения локальных пересборок.
# Объединяем apt-get update и install, очищаем кэш apt.
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 git && \
    rm -rf /var/lib/apt/lists/* # Очищаем кэш apt для уменьшения размера образа

# Копируем остальное содержимое приложения (включая main.py, detector.py, templates/)
# Этот шаг будет кэшироваться, если только код приложения не изменился.
COPY . .

# Обязательно для YOLO, чтобы найти модель. Доступ к этой папке будет через volume.
ENV YOLO_MODELS_DIR=/shared_data/models

# Порт, который будет слушать Flask
EXPOSE 5000

# Запускаем приложение
CMD ["python", "main.py"]