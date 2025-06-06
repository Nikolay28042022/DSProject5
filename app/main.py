# app/main.py

import os
import threading
import asyncio
import queue
import time
import cv2


# --- Импорты для Flask ---
from flask import Flask, Response, render_template

# --- Импорты для Telegram Bot ---
from telegram import Bot
from telegram.error import TelegramError
import config # config.py находится в той же папке

# --- Импорт нашего нового модуля детектирования ---
import detector as det # Сократим алиас до det

# --- Импорты для gTTS (Text-to-Speech) ---
from gtts import gTTS
import tempfile

# --- Конфигурация для main.py ---
# Эти переменные будут переданы в det.start_video_detection.
# Пути теперь относительны к корневой папке 'app' внутри контейнера.
VIDEO_PATH = 'videos/video.mp4'
MIN_AREA = 500
PHOTO_INTERVAL = 5
OUTPUT_FOLDER = 'output' # Эта папка будет внутри /app/output в контейнере

# Инициализация Telegram Bot
telegram_bot = Bot(token=config.TELEGRAM_BOT_TOKEN)

# --- Flask приложение ---
app = Flask(__name__, template_folder='templates') # Указываем путь к templates явно

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """
    Функция-генератор, которая читает кадры из detector.py
    и форматирует их для MJPEG-стриминга.
    """
    while True:
        frame = det.get_current_frame_for_stream() # Получаем кадр из detector
        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


# --- Асинхронные функции для отправки сообщений в Telegram ---
async def send_voice_notification(chat_id, text):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='ru', slow=False)
            tts.save(fp.name)
            temp_audio_path = fp.name

        with open(temp_audio_path, 'rb') as audio_file:
            await telegram_bot.send_audio(chat_id=chat_id, audio=audio_file)
        
        print(f"[Telegram Voice] Голосовое сообщение отправлено: '{text[:50]}...'")

    except Exception as e:
        print(f"[Telegram Voice ERROR] Не удалось отправить голосовое сообщение: {e}")
    finally:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

async def send_telegram_notification_async(photo_path, message_text, web_link, voice_text=None):
    try:
        with open(photo_path, 'rb') as photo_file:
            await telegram_bot.send_photo(chat_id=config.TELEGRAM_CHAT_ID, photo=photo_file)

        await telegram_bot.send_message(
            chat_id=config.TELEGRAM_CHAT_ID,
            text=f"{message_text}\n\nПосмотреть видеопоток: <a href='{web_link}'>Нажмите здесь</a>",
            parse_mode='HTML',
            disable_web_page_preview=True
        )
        print(f"[Telegram] Текстовое уведомление отправлено в Telegram: {message_text}")

        if voice_text:
            await send_voice_notification(config.TELEGRAM_CHAT_ID, voice_text)

    except TelegramError as e:
        print(f"[Telegram ERROR] Не удалось отправить уведомление в Telegram: {e}")
    except FileNotFoundError:
        print(f"[Telegram ERROR] Файл фотографии не найден для отправки в Telegram: {photo_path}")
    except Exception as e:
        print(f"[Telegram ERROR] Неизвестная ошибка при отправке в Telegram: {e}")

# --- Поток для Telegram-бота (теперь читает из очереди detector) ---
def telegram_bot_thread_run():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("[Telegram Thread] Telegram-бот запущен и ожидает задач из Detector.")

    async def process_queue():
        while True:
            try:
                # Читаем из очереди det.motion_detection_queue
                task_data = det.motion_detection_queue.get(timeout=1)
                
                photo_path = task_data['photo_path']
                message_text = task_data['message_text']
                web_link = config.WEB_SERVER_URL
                voice_text = task_data['voice_text']

                await send_telegram_notification_async(photo_path, message_text, web_link, voice_text)
                det.motion_detection_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[Telegram Thread ERROR] Ошибка при обработке задачи из очереди: {e}")

    loop.run_until_complete(process_queue())


# --- Основная точка входа ---
if __name__ == "__main__":
    # Убедимся, что папка output существует внутри app/
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Папка collected_images будет смонтирована извне.
    # Создавать ее здесь в коде контейнера не нужно, Docker сделает это за нас.
    # Но для локального запуска без Docker, если тестировать, можно оставить:
    # if not os.path.exists(os.path.join(os.getcwd(), 'shared_data', 'collected_images')):
    #    os.makedirs(os.path.join(os.getcwd(), 'shared_data', 'collected_images'))


    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == 'ВАШ_ТОКЕН_БОТА':
        print("ERROR: Telegram Bot Token не настроен в config.py!")
        print("Пожалуйста, получите токен от @BotFather и обновите config.py.")
        exit(1)
    if not config.TELEGRAM_CHAT_ID or config.TELEGRAM_CHAT_ID == 'ВАШ_ID_ЧАТА':
         print("ERROR: Telegram Chat ID не настроен в config.py!")
         print("Пожалуйста, получите Chat ID и обновите config.py.")
         exit(1)

    # --- Запуск потока детектирования из detector.py ---
    detector_thread = threading.Thread(
        target=det.start_video_detection,
        # Путь к видео теперь 'videos/video.mp4' внутри app/videos
        # output_folder - это 'output/' внутри app/output
        args=(VIDEO_PATH, MIN_AREA, PHOTO_INTERVAL, True, OUTPUT_FOLDER) # collect_images_mode=True
    )
    detector_thread.daemon = True
    detector_thread.start()

    telegram_thread = threading.Thread(target=telegram_bot_thread_run)
    telegram_thread.daemon = True
    telegram_thread.start()

    print(f"Запуск Flask-приложения на http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}/")
    app.run(host=config.WEB_SERVER_HOST, port=config.WEB_SERVER_PORT, debug=False, use_reloader=False)

    detector_thread.join()
    print("Приложение завершило работу.")
