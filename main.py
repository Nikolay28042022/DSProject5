# main.py

import cv2
import imutils
import os
import time
import threading
import datetime
import asyncio
import queue
from io import BytesIO

# --- Импорты для Flask ---
from flask import Flask, Response, render_template

# --- Импорты для Telegram Bot ---
from telegram import Bot
from telegram.error import TelegramError
import config

# --- Импорты для YOLO ---
from ultralytics import YOLO
import torch

# --- Импорты для gTTS (Text-to-Speech) ---
from gtts import gTTS
import tempfile # Для создания временных файлов

# --- Конфигурация ---
VIDEO_PATH = 'videos/video.mp4'
MIN_AREA = 500
PHOTO_INTERVAL = 5
OUTPUT_FOLDER = 'output'

# Глобальные переменные для обмена данными между потоками
current_frame = None
frame_lock = threading.Lock()

# Очередь для Telegram-уведомлений
telegram_queue = queue.Queue()

# Инициализация Telegram Bot
telegram_bot = Bot(token=config.TELEGRAM_BOT_TOKEN)

# --- Инициализация YOLO модели ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[YOLO] Используется устройство: {DEVICE}")

yolo_model = YOLO('yolov8n.pt')
print("[YOLO] Модель YOLOv8n успешно загружена.")


# --- Flask приложение ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global current_frame, frame_lock
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue

            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

# --- НОВАЯ ФУНКЦИЯ: Генерация и отправка голосового сообщения ---
async def send_voice_notification(chat_id, text):
    """
    Генерирует голосовое сообщение из текста и отправляет его в Telegram.
    """
    try:
        # Создаем временный файл для сохранения аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='ru', slow=False) # lang='ru' для русского языка
            tts.save(fp.name)
            temp_audio_path = fp.name

        # Отправляем аудиофайл
        with open(temp_audio_path, 'rb') as audio_file:
            await telegram_bot.send_audio(chat_id=chat_id, audio=audio_file)
        
        print(f"[Telegram Voice] Голосовое сообщение отправлено: '{text[:50]}...'")

    except Exception as e:
        print(f"[Telegram Voice ERROR] Не удалось отправить голосовое сообщение: {e}")
    finally:
        # Удаляем временный файл
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- Обновленная асинхронная функция для отправки сообщений в Telegram ---
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

        # Если есть текст для голоса, отправляем голосовое сообщение
        if voice_text:
            await send_voice_notification(config.TELEGRAM_CHAT_ID, voice_text)

    except TelegramError as e:
        print(f"[Telegram ERROR] Не удалось отправить уведомление в Telegram: {e}")
    except FileNotFoundError:
        print(f"[Telegram ERROR] Файл фотографии не найден для отправки в Telegram: {photo_path}")
    except Exception as e:
        print(f"[Telegram ERROR] Неизвестная ошибка при отправке в Telegram: {e}")

# --- Поток для Telegram-бота (изменена обработка очереди) ---
def telegram_bot_thread_run():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print("[Telegram Thread] Telegram-бот запущен и ожидает задач.")

    async def process_queue():
        while True:
            try:
                # Задача теперь может включать текст для голосового сообщения
                task_data = telegram_queue.get(timeout=1)
                photo_path, message_text, web_link, voice_text = task_data # Распаковываем 4 элемента
                await send_telegram_notification_async(photo_path, message_text, web_link, voice_text)
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[Telegram Thread ERROR] Ошибка при обработке задачи из очереди: {e}")

    loop.run_until_complete(process_queue())

# --- Логика обработки видеопотока ---
def video_processing_thread():
    global current_frame, frame_lock

    print(f"[Video Processing] Попытка открыть видеофайл: {VIDEO_PATH}")

    if not os.path.exists(VIDEO_PATH):
        print(f"[Video Processing] Ошибка: Файл не найден по пути: {VIDEO_PATH}")
        print("[Video Processing] Пожалуйста, убедитесь, что 'video.mp4' находится в папке 'DSProject5/videos/'")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[Video Processing] Ошибка: Не удалось открыть видеофайл {VIDEO_PATH}")
        return

    print("[Video Processing] Видеофайл успешно открыт. Нажмите 'q' для выхода из основного окна.")

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    last_photo_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[Video Processing] Конец видеофайла. Перезапуск видео для демонстрации.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = imutils.resize(frame, width=800)
        original_frame_copy = frame.copy()

        detected_objects_names = []
        # Запускаем YOLO только на кадрах, где есть движение (или просто периодически)
        # Для демонстрации голосовых сообщений пока запустим YOLO на всех кадрах
        # Или можно оставить условие motion_detected_mog2 и/или time.time() - last_photo_time >= PHOTO_INTERVAL
        
        # Для тестов YOLO можно временно убрать if, чтобы он работал на каждом кадре
        # if motion_detected_mog2 or (current_time - last_photo_time >= PHOTO_INTERVAL): # Добавим или, чтобы не пропустить
        #    pass # Но лучше просто сделать, чтобы YOLO запускался, если есть движение MOG2
        
        # Запускаем YOLO для классификации, если MOG2 обнаружил движение, или если прошло достаточно времени
        yolo_run_for_telegram = False # Флаг, был ли YOLO запущен для генерации Telegram-сообщения
        
        current_time = time.time()
        
        # Сначала обрабатываем MOG2 для определения движения и триггера Telegram
        motion_detected_mog2 = False
        gray = cv2.cvtColor(original_frame_copy, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgmask = fgbg.apply(gray)
        thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < MIN_AREA:
                continue
            motion_detected_mog2 = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Зеленая рамка MOG2


        # --- YOLO ОБНАРУЖЕНИЕ ОБЪЕКТОВ ---
        # Запускаем YOLO только если было движение MOG2 ИЛИ если прошло достаточно времени для нового уведомления
        if motion_detected_mog2 and (current_time - last_photo_time >= PHOTO_INTERVAL):
            yolo_run_for_telegram = True
            results = yolo_model(original_frame_copy, conf=0.5, verbose=False, device=DEVICE)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)
                    cls = int(box.cls[0])
                    name = yolo_model.names[cls]

                    # Рисуем рамку и текст на основном кадре (frame) для YOLO
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Синий цвет рамки YOLO
                    text = f"{name} {conf}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    detected_objects_names.append(name)
        
        # Обновляем current_frame для веб-стриминга (теперь он с рамками MOG2 и YOLO)
        with frame_lock:
            current_frame = frame.copy()


        if motion_detected_mog2 and (current_time - last_photo_time >= PHOTO_INTERVAL):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            photo_filename = f"motion_detected_{timestamp}.jpg"
            full_photo_path = os.path.join(OUTPUT_FOLDER, photo_filename)

            cv2.imwrite(full_photo_path, frame) # Сохраняем кадр с рамками MOG2 и YOLO

            print(f"[Video Processing] Движение обнаружено! Сохранено фото: {photo_filename}")

            # Формируем текстовое сообщение для Telegram
            message_parts = []
            message_parts.append("Обнаружено движение!")
            
            voice_message_text = "Обнаружено движение. " # Начало голосового сообщения

            if detected_objects_names:
                unique_objects = ", ".join(sorted(list(set(detected_objects_names))))
                message_parts.append(f"Объекты: {unique_objects}")
                voice_message_text += f"Объекты: {unique_objects}."
            else:
                message_parts.append("Объекты не классифицированы.")
                voice_message_text += "Объекты не классифицированы." # Для голоса тоже

            message_text_telegram = f"{' '.join(message_parts)} в {datetime.datetime.now().strftime('%H:%M:%S')}!"
            
            # --- НОВАЯ ЧАСТЬ: Передача задачи в очередь Telegram-бота с текстом для голоса ---
            telegram_queue.put((full_photo_path, message_text_telegram, config.WEB_SERVER_URL, voice_message_text))
            # --- КОНЕЦ НОВОЙ ЧАСТИ ---

            last_photo_time = current_time

        cv2.imshow("Video Stream (with YOLO)", frame)
        cv2.imshow("Threshold (Motion Mask)", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Video Processing] Поток обработки видео завершил работу.")


# --- Основная точка входа ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == 'ВАШ_ТОКЕН_БОТА':
        print("ERROR: Telegram Bot Token не настроен в config.py!")
        print("Пожалуйста, получите токен от @BotFather и обновите config.py.")
        exit(1)
    if not config.TELEGRAM_CHAT_ID or config.TELEGRAM_CHAT_ID == 'ВАШ_ID_ЧАТА':
         print("ERROR: Telegram Chat ID не настроен в config.py!")
         print("Пожалуйста, получите Chat ID и обновите config.py.")
         exit(1)

    telegram_thread = threading.Thread(target=telegram_bot_thread_run)
    telegram_thread.daemon = True
    telegram_thread.start()

    video_thread = threading.Thread(target=video_processing_thread)
    video_thread.daemon = True
    video_thread.start()

    print(f"Запуск Flask-приложения на http://{config.WEB_SERVER_HOST}:{config.WEB_SERVER_PORT}/")
    app.run(host=config.WEB_SERVER_HOST, port=config.WEB_SERVER_PORT, debug=False, use_reloader=False)

    video_thread.join()
    print("Приложение завершило работу.")
