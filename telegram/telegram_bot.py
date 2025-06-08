import telebot
import os
import queue
import threading
import time
from gtts import gTTS
from dotenv import load_dotenv
from flask import Flask, request, jsonify # Импортируем Flask для API

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен бота и ID чата из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_FLASK_PORT = os.getenv('TELEGRAM_FLASK_PORT', 5001) # Новый порт для Flask API бота

# Проверяем, что токен и ID чата установлены
if not TELEGRAM_BOT_TOKEN:
    print("[Telegram Bot ERROR] Токен Telegram-бота не установлен. Убедитесь, что TELEGRAM_BOT_TOKEN присутствует в файле .env")
    exit()
if not TELEGRAM_CHAT_ID:
    print("[Telegram Bot ERROR] ID чата Telegram не установлен. Убедитесь, что TELEGRAM_CHAT_ID присутствует в файле .env")
    exit()

# Инициализируем бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Очередь для задач Telegram (фото, текст, голос)
telegram_task_queue = queue.Queue()

def send_message_with_photo_and_voice(photo_path, message_text, voice_text):
    """
    Отправляет сообщение с фото и голосовым сообщением в указанный Telegram-чат.
    :param photo_path: Путь к файлу фотографии.
    :param message_text: Текст сообщения.
    :param voice_text: Текст для голосового сообщения.
    """
    try:
        # Отправляем фотографию
        if os.path.exists(photo_path):
            with open(photo_path, 'rb') as photo:
                bot.send_photo(TELEGRAM_CHAT_ID, photo, caption=message_text)
            print(f"[Telegram] Фото уведомление отправлено: {photo_path}")
            # Удаляем фото после отправки, чтобы не заполнять диск (опционально, если они не нужны для отладки)
            # os.remove(photo_path)
        else:
            print(f"[Telegram ERROR] Файл фото не найден: {photo_path}. Отправляю только сообщение.")
            bot.send_message(TELEGRAM_CHAT_ID, f"Обнаружено движение, но фото не найдено: {message_text}")

        # Генерируем голосовое сообщение
        if voice_text:
            tts = gTTS(text=voice_text, lang='ru')
            voice_filename = "voice_message.ogg"
            # Сохраняем голосовое сообщение во временной папке /app/output, так как она монтируется из shared_data
            voice_path = os.path.join('/app/output', voice_filename) # Сохраняем в output, чтобы было доступно в обоих контейнерах
            tts.save(voice_path)
            
            # Отправляем голосовое сообщение
            if os.path.exists(voice_path):
                with open(voice_path, 'rb') as voice:
                    bot.send_voice(TELEGRAM_CHAT_ID, voice)
                print(f"[Telegram Voice] Голосовое сообщение отправлено: {voice_text}")
                # Удаляем голосовое сообщение после отправки (опционально)
                os.remove(voice_path)
            else:
                print(f"[Telegram Voice ERROR] Не удалось создать голосовой файл: {voice_path}")
        
    except Exception as e:
        print(f"[Telegram ERROR] Ошибка при отправке сообщения в Telegram: {e}")

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я бот для обнаружения движения. Я буду отправлять вам уведомления с фотографиями и голосовыми сообщениями при обнаружении активности.")
    print(f"[Telegram Bot] Получена команда /start от пользователя {message.chat.id}")

# --- API для приема задач от детектора ---
telegram_api_app = Flask(__name__)

@telegram_api_app.route('/send_task', methods=['POST'])
def receive_task():
    """
    Принимает задачу от детектора (сервиса app) и добавляет ее в очередь.
    Ожидает JSON вида: {"photo_path": "...", "message_text": "...", "voice_text": "..."}
    """
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data received")

        photo_path = data.get('photo_path')
        message_text = data.get('message_text')
        voice_text = data.get('voice_text')

        if not photo_path or not message_text:
            raise ValueError("Missing required fields: photo_path or message_text")

        telegram_task_queue.put({
            'photo_path': photo_path,
            'message_text': message_text,
            'voice_text': voice_text
        })
        print(f"[Telegram API] Задача получена и добавлена в очередь: {photo_path}")
        return jsonify({"status": "success", "message": "Task added to queue"}), 200
    except Exception as e:
        print(f"[Telegram API ERROR] Ошибка при получении задачи: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# Функция для запуска Flask API в отдельном потоке
def run_flask_api():
    print(f"[Telegram API] Запуск Flask API на 0.0.0.0:{TELEGRAM_FLASK_PORT}")
    telegram_api_app.run(host='0.0.0.0', port=int(TELEGRAM_FLASK_PORT), debug=False, use_reloader=False)

# Запуск обработки задач из очереди Telegram-бота
def telegram_queue_processor():
    """
    Постоянно проверяет очередь задач и отправляет сообщения в Telegram.
    """
    while True:
        if not telegram_task_queue.empty():
            task = telegram_task_queue.get()
            print(f"[Telegram Processor] Обработка задачи из очереди: {task['photo_path']}")
            send_message_with_photo_and_voice(
                task['photo_path'],
                task['message_text'],
                task['voice_text']
            )
            telegram_task_queue.task_done()
        time.sleep(0.1)

# Главная точка входа для Telegram-бота
if __name__ == '__main__':
    # Запускаем Flask API в отдельном потоке
    flask_thread = threading.Thread(target=run_flask_api, daemon=True)
    flask_thread.start()

    # Запускаем обработчик очереди Telegram в отдельном потоке
    processor_thread = threading.Thread(target=telegram_queue_processor, daemon=True)
    processor_thread.start()
    print("[Telegram Main] Telegram-бот и API запущены. Ожидают задач и сообщений.")

    # Запускаем polling для получения сообщений от Telegram API (например, команды /start)
    try:
        print("[Telegram Polling] Бот начинает опрос новых сообщений...")
        bot.polling(none_stop=True, interval=0, timeout=20)
    except Exception as e:
        print(f"[Telegram Polling ERROR] Ошибка при запуске polling бота: {e}")

