# Dockerfile.app

# Используем базовый образ Python (например, slim-buster для меньшего размера)
FROM python:3.10-slim-buster

# Устанавливаем необходимые системные зависимости для OpenCV
# и curl (для загрузки моделей YOLO, если их нет локально)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем все из папки 'app' в рабочую директорию контейнера
COPY app/ .


# Копируем папку shared_data (или, по крайней мере, ее структуру)
# На этапе сборки, чтобы пути внутри контейнера были корректными.
# Фактические данные будут монтироваться через volume.
RUN mkdir -p shared_data/collected_images shared_data/models

# Устанавливаем Python-зависимости из requirements_app.txt
RUN pip install --no-cache-dir -r requirements_app.txt

# Специальная установка torch, torchvision, torchaudio для CPU
# Важно: используйте версии, которые гарантированно совместимы с Python 3.10 и CPU
# Пожалуйста, проверьте актуальные версии на https://pytorch.org/get-started/locally/
# Выбирайте: Linux, Pip, Python 3.10, CPU
# На основе данных с PyTorch.org (сегодня, 2025-06-05), для Py3.10 CPU, это 2.3.1
RUN pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем ultralytics (она будет использовать уже установленный torch)
# Укажите здесь вашу версию ultralytics, например 8.1.0
RUN pip install --no-cache-dir ultralytics==8.1.0

# Открываем порт для Flask-приложения
EXPOSE 5000

# Команда для запуска приложения
# Python main.py будет выполнен при запуске контейнера
CMD ["python", "main.py"]
