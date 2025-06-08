# app/detector.py

import cv2
import imutils
import os
import time
import threading
import datetime
# import queue # Больше не нужна внутренняя очередь, т.к. отправляем по HTTP
import requests # Импортируем для HTTP-запросов
import json # Для работы с JSON-данными для HTTP-запроса

from ultralytics import YOLO
import torch

# --- Конфигурация для модуля детектирования ---
VIDEO_PATH = 'videos/video.mp4' # Внутри контейнера app

MIN_AREA = 1000 # Увеличен до 1000, чтобы игнорировать мелкое движение
DETECTION_INTERVAL_SECONDS = 3.0 # Как часто запускать YOLO для визуализации на стриме (увеличено)
COLLECT_IMAGE_INTERVAL_SECONDS = 60 # Как часто сохранять кадры для разметки при движении (увеличено)

# Глобальные переменные для обмена данными внутри модуля или с main.py
current_frame_for_stream = None # Кадр для веб-стриминга (с рамками)
raw_frame_for_collection = None # Сырой кадр для сбора (без рамок)
detector_lock = threading.Lock() # Мьютекс для безопасного доступа к кадрам

# Инициализация YOLO модели
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[YOLO Detector] Используется устройство: {DEVICE}")
yolo_model_path = os.path.join('shared_data', 'models', 'yolov8n.pt') # Путь внутри контейнера
yolo_model = YOLO(yolo_model_path)
print("[YOLO Detector] Модель YOLOv8n успешно загружена.")

# Очередь для передачи информации о движении/классификации в main.py
# motion_detection_queue = queue.Queue() # Больше не нужна

# --- Флаг для режима сбора изображений ---
COLLECT_IMAGES_MODE = False
COLLECTED_IMAGES_BASE_FOLDER = os.path.join('shared_data', 'collected_images')
COLLECTED_IMAGES_CURRENT_RUN_FOLDER = None # Будет инициализирован при запуске потока
last_collection_time = time.time()


# --- Главный поток обработки видео ---
def start_video_detection(video_path, min_area, telegram_photo_interval,
                           collect_images_mode=False,
                           output_folder='output',
                           telegram_api_url='http://telegram:5001/send_task'): # Новый параметр для URL Telegram API
    """
    Запускает поток обработки видео.
    :param video_path: Путь к видеофайлу.
    :param min_area: Минимальная площадь для обнаружения движения.
    :param telegram_photo_interval: Интервал для отправки снимков в Telegram.
    :param collect_images_mode: Если True, сохраняет кадры для разметки.
    :param output_folder: Папка для сохранения обнаруженных снимков (внутри контейнера app).
    :param telegram_api_url: URL для отправки задач Telegram-боту.
    """
    global current_frame_for_stream, detector_lock
    global raw_frame_for_collection, COLLECT_IMAGES_MODE, COLLECTED_IMAGES_CURRENT_RUN_FOLDER, last_collection_time
    global DETECTION_INTERVAL_SECONDS, COLLECT_IMAGE_INTERVAL_SECONDS # Для доступа к переменным внутри функции

    COLLECT_IMAGES_MODE = collect_images_mode
    if COLLECT_IMAGES_MODE:
        COLLECTED_IMAGES_CURRENT_RUN_FOLDER = os.path.join(COLLECTED_IMAGES_BASE_FOLDER, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(COLLECTED_IMAGES_CURRENT_RUN_FOLDER):
            os.makedirs(COLLECTED_IMAGES_CURRENT_RUN_FOLDER)
            print(f"[Detector] Режим сбора изображений активирован. Изображения будут сохраняться в: {COLLECTED_IMAGES_CURRENT_RUN_FOLDER}")
        else:
            print(f"[Detector] Режим сбора изображений активирован. Изображения будут сохраняться в существующую папку: {COLLECTED_IMAGES_CURRENT_RUN_FOLDER}")


    print(f"[Detector] Попытка открыть видеофайл: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[Detector ERROR] Ошибка: Файл не найден по пути: {video_path}")
        print("[Detector ERROR] Пожалуйста, убедитесь, что 'video.mp4' находится в папке 'app/videos/' внутри контейнера.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Detector ERROR] Не удалось открыть видеофайл {video_path}")
        return

    print("[Detector] Видеофайл успешно открыт.")

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    last_telegram_photo_time = time.time()
    last_yolo_run_time = time.time()

    while True:
        start_loop_time = time.time() # Начало итерации цикла

        ret, frame = cap.read()

        if not ret:
            print("[Detector] Конец видеофайла. Перезапуск видео для демонстрации.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = imutils.resize(frame, width=480) # Уменьшен размер кадра до 480
        original_frame_copy = frame.copy()

        detected_objects_names = []
        motion_detected_mog2 = False

        # --- MOG2: Обнаружение движения ---
        start_mog2_time = time.time() # Тайминг
        gray = cv2.cvtColor(original_frame_copy, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgmask = fgbg.apply(gray)
        thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            motion_detected_mog2 = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Зеленая рамка MOG2
        end_mog2_time = time.time() # Тайминг
        # print(f"[Detector Timing] MOG2 Processing: {end_mog2_time - start_mog2_time:.4f}s")


        # --- YOLO: Обнаружение объектов ---
        current_time = time.time()
        # Запускаем YOLO, если есть движение MOG2, ИЛИ если прошло достаточно времени с последнего запуска
        # Это позволяет YOLO не "спать" слишком долго, даже если MOG2 не видит движения.
        if motion_detected_mog2 or (current_time - last_yolo_run_time >= DETECTION_INTERVAL_SECONDS):
            start_yolo_time = time.time() # Тайминг
            results = yolo_model(original_frame_copy, conf=0.5, verbose=False, device=DEVICE)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = round(float(box.conf[0]), 2)
                    cls = int(box.cls[0])
                    name = yolo_model.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Синяя рамка YOLO
                    text = f"{name} {conf}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    detected_objects_names.append(name)
            end_yolo_time = time.time() # Тайминг
            last_yolo_run_time = current_time # Обновляем время последнего запуска YOLO
            # print(f"[Detector Timing] YOLO Detection: {end_yolo_time - start_yolo_time:.4f}s")


        # --- Обновление кадра для веб-стриминга ---
        with detector_lock:
            start_copy_time = time.time() # Тайминг
            current_frame_for_stream = frame.copy() # Кадр с обеими рамками
            raw_frame_for_collection = original_frame_copy.copy() # Сырой кадр для сбора
            end_copy_time = time.time() # Тайминг
            # print(f"[Detector Timing] Frame Copy: {end_copy_time - start_copy_time:.4f}s")


        # --- Логика отправки в Telegram (через HTTP-запрос) ---
        if motion_detected_mog2 and (current_time - last_telegram_photo_time >= telegram_photo_interval):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            photo_filename = f"motion_detected_{timestamp}.jpg"
            # Сохраняем фото в /app/output, так как эта папка монтируется в контейнерах
            full_photo_path = os.path.join(output_folder, photo_filename) 

            cv2.imwrite(full_photo_path, frame) # Сохраняем кадр с рамками

            message_parts = ["Обнаружено движение!"]
            voice_message_text = "Обнаружено движение. "

            if detected_objects_names:
                unique_objects = ", ".join(sorted(list(set(detected_objects_names))))
                message_parts.append(f"Объекты: {unique_objects}")
                voice_message_text += f"Объекты: {unique_objects}."
            else:
                message_parts.append("Объекты не классифицированы.")
                voice_message_text += "Объекты не классифицированы."

            message_text_telegram = f"{' '.join(message_parts)} в {datetime.datetime.now().strftime('%H:%M:%S')}!"
            
            # Отправляем HTTP POST-запрос на Telegram API
            try:
                payload = {
                    'photo_path': full_photo_path,
                    'message_text': message_text_telegram,
                    'voice_text': voice_message_text
                }
                # Имя сервиса 'telegram' резолвится в его IP внутри Docker-сети
                response = requests.post(telegram_api_url, json=payload, timeout=5)
                response.raise_for_status() # Вызывает исключение для HTTP ошибок (4xx или 5xx)
                print(f"[Detector] Задача для Telegram успешно отправлена по HTTP: {response.json()}")
            except requests.exceptions.RequestException as e:
                print(f"[Detector ERROR] Ошибка при отправке задачи в Telegram API: {e}")

            last_telegram_photo_time = current_time
            print(f"[Detector] Задача для Telegram добавлена в очередь: {full_photo_path}")


        # --- Режим сбора изображений для разметки ---
        if COLLECT_IMAGES_MODE and motion_detected_mog2 and (current_time - last_collection_time >= COLLECT_IMAGE_INTERVAL_SECONDS):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            collection_photo_filename = f"collected_{timestamp}.jpg"
            full_collection_photo_path = os.path.join(COLLECTED_IMAGES_CURRENT_RUN_FOLDER, collection_photo_filename)
            
            cv2.imwrite(full_collection_photo_path, raw_frame_for_collection)
            print(f"[Detector] Кадр сохранен для разметки: {full_collection_photo_path}")
            last_collection_time = current_time

        # --- Конец итерации цикла ---
        end_loop_time = time.time() # Тайминг
        frame_processing_time = end_loop_time - start_loop_time
        # print(f"[Detector Timing] Total Frame Processing: {frame_processing_time:.4f}s")
        # print(f"[Detector Timing] Approximate FPS: {1 / frame_processing_time:.2f} FPS") # Тайминг

    cap.release()
    print("[Detector] Поток обработки видео завершил работу.")

# --- Вспомогательные функции для Flask ---
def get_current_frame_for_stream():
    """Возвращает текущий кадр для веб-стриминга."""
    with detector_lock:
        return current_frame_for_stream.copy() if current_frame_for_stream is not None else None
