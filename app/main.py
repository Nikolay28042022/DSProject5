import os
import threading
import time
import cv2
from flask import Flask, Response, render_template
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Импортируем модуль detector
import detector

# Конфигурация приложения
WEB_SERVER_URL = os.getenv('WEB_SERVER_URL', 'http://127.0.0.1:5000/') # Получаем URL из .env
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_FLASK_PORT = os.getenv('TELEGRAM_FLASK_PORT', 5001) # Порт API Telegram-бота

# Параметры детектирования
VIDEO_SOURCE = 'videos/video.mp4'  # Путь к видеофайлу внутри контейнера
MIN_AREA_FOR_MOTION = 1000          # Минимальная площадь для обнаружения движения
TELEGRAM_PHOTO_INTERVAL = 10        # Интервал отправки фото в Telegram (секунды)
COLLECT_IMAGES_FOR_TRAINING = False # Активировать режим сбора изображений
OUTPUT_FOLDER = 'output'            # Папка для сохранения фото, отправляемых в Telegram

# Создаем Flask-приложение
app = Flask(__name__)

# Создаем папку output, если она не существует
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"[Main App] Создана папка для выходных данных: {OUTPUT_FOLDER}")

# Инициализация и запуск потока детектора
# Передаем TELEGRAM_API_URL и WEB_SERVER_URL
telegram_api_url_internal = f"http://telegram:{TELEGRAM_FLASK_PORT}/send_task" # URL для обращения к Telegram-боту внутри Docker-сети

detector_thread = threading.Thread(target=detector.start_video_detection,
                                   args=(VIDEO_SOURCE, MIN_AREA_FOR_MOTION, TELEGRAM_PHOTO_INTERVAL,
                                         COLLECT_IMAGES_FOR_TRAINING, OUTPUT_FOLDER,
                                         telegram_api_url_internal, WEB_SERVER_URL), # Передаем WEB_SERVER_URL
                                   daemon=True)
detector_thread.start()
print("[Main App] Поток детектора запущен.")

# ====================================================================================
# Flask-роуты для веб-интерфейса
# ====================================================================================

@app.route('/')
def index():
    """Главная страница веб-приложения."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Видеопоток для отображения в браузере."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """
    Генератор кадров для видеопотока.
    Получает текущий кадр от детектора и кодирует его в JPEG.
    """
    while True:
        # Получаем текущий кадр от детектора
        frame = detector.get_current_frame_for_stream()

        if frame is not None:
            # Кодируем кадр в JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                # Отправляем кадр в формате multipart/x-mixed-replace
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print("[Main App ERROR] Ошибка кодирования кадра в JPEG.")
        
        # Небольшая задержка, чтобы не нагружать CPU слишком сильно, если кадры идут очень быстро
        time.sleep(0.01) # Например, 10 мс

# ====================================================================================
# Запуск Flask-приложения
# ====================================================================================
if __name__ == '__main__':
    print(f"Запуск Flask-приложения на {WEB_SERVER_URL}")
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False для продакшн-окружения
