# config.py

# --- Telegram Bot Configuration ---
TELEGRAM_BOT_TOKEN = '7747206707:AAHqOBDBo3L7_dOtWfO5ldpPXpUp2YtDbns' # Замените на токен, полученный от BotFather
TELEGRAM_CHAT_ID = '690533661'     # Замените на ID вашего чата (число)
                                     # Если это группа/канал, ID будет отрицательным (например, -1234567890)

# --- Flask Web Server Configuration (можно оставить здесь или перенести сюда из main.py) ---
WEB_SERVER_HOST = '0.0.0.0'
WEB_SERVER_PORT = 5000
WEB_SERVER_URL = f"http://192.168.0.222:{WEB_SERVER_PORT}" # Ссылка для Telegram (если веб-сервер доступен извне)
                                                      # Для локальной разработки используем 127.0.0.1
