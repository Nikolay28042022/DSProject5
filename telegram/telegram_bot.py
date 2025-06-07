import telebot
import os
import queue
import threading
import time
from gtts import gTTS
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем токен бота и ID чата из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

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
            # Удаляем фото после отправки, чтобы не заполнять диск (опционально)
            # os.remove(photo_path)
        else:
            print(f"[Telegram ERROR] Файл фото не найден: {photo_path}")
            bot.send_message(TELEGRAM_CHAT_ID, f"Обнаружено движение, но фото не найдено: {message_text}")

        # Генерируем голосовое сообщение
        if voice_text:
            tts = gTTS(text=voice_text, lang='ru')
            voice_filename = "voice_message.ogg"
            # Сохраняем голосовое сообщение во временной папке или рядом с основным скриптом
            # Важно: В Docker контейнере пути могут быть специфичными, убедитесь, что /app доступен для записи
            voice_path = os.path.join('/app', voice_filename) 
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

# Функция для обработки задач из очереди
def telegram_task_processor():
    """
    Постоянно проверяет очередь задач и отправляет сообщения в Telegram.
    """
    while True:
        if not telegram_task_queue.empty():
            task = telegram_task_queue.get()
            print(f"[Telegram Bot] Обработка задачи из очереди: {task['photo_path']}")
            send_message_with_photo_and_voice(
                task['photo_path'],
                task['message_text'],
                task['voice_text']
            )
            telegram_task_queue.task_done() # Сообщаем, что задача выполнена
        time.sleep(0.1) # Небольшая задержка, чтобы не нагружать CPU

# Запуск обработки задач в отдельном потоке
telegram_processor_thread = threading.Thread(target=telegram_task_processor, daemon=True)
telegram_processor_thread.start()
print("[Telegram Thread] Telegram-бот запущен и ожидает задач из Detector.")

# Запуск бота (должен быть в основном потоке или его нужно запускать в отдельном потоке без daemon=True)
# В нашем случае, так как это CMD["python", "telegram_bot.py"], это будет основной поток контейнера
if __name__ == '__main__':
    try:
        print("[Telegram Bot] Бот начинает опрос новых сообщений...")
        bot.polling(none_stop=True, interval=0, timeout=20)
    except Exception as e:
        print(f"[Telegram Bot ERROR] Ошибка при запуске polling бота: {e}")