# Используем базовый образ Python
FROM python:3.10-slim-buster

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальное содержимое приложения (ваш телеграм-бот)
COPY . .

# Запускаем скрипт телеграм-бота
CMD ["python", "telegram_bot.py"]