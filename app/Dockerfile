    # Используем официальный образ PyTorch с поддержкой CUDA.
    FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

    # Устанавливаем рабочую директорию
    WORKDIR /app

    # Копируем .env файл в рабочую директорию контейнера
    COPY .env .
    # Копируем requirements.txt отдельно, чтобы изменения в коде приложения
    # не приводили к переустановке зависимостей при каждом билде.
    COPY app/requirements.txt . 

    # Устанавливаем зависимости Python и системные пакеты.
    RUN pip install -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu118 && \
        apt-get update && \
        apt-get install -y ffmpeg libsm6 libxext6 git && \
        rm -rf /var/lib/apt/lists/*

    # Копируем остальное содержимое приложения (main.py, detector.py, templates/)
    COPY app/ . 

    # Обязательно для YOLO, чтобы найти модель. Доступ к этой папке будет через volume.
    ENV YOLO_MODELS_DIR=/shared_data/models

    # Порт, который будет слушать Flask
    EXPOSE 5000

    # Запускаем приложение
    CMD ["python", "main.py"]
    