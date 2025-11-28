# TravelMate Chat Bot

Telegram-бот для туристических вопросов по городам и туристическим направлениям Азии (Токио, Сеул, Бангкок, Сингапур, Пхукет, Фукуок, Бали, Хиккадува).
Бот отвечает на вопросы о визах, экскурсиях, обмене валют, транспорте и других туристических темах.

Telegram: [@TravelMateXBot](https://t.me/TravelMateXBot)

---

## Структура проекта

```
.
├── config.py               # Конфигурация модели (устройство, размеры слоев, learning rate)
├── nltk_utils.py           # Токенизация, стемминг, Bag-of-Words
├── model.py                # Определение нейросети
├── train.py                # Скрипт обучения модели
├── telegram_bot.py         # Основной код Telegram-бота
├── cities.json             # Список городов и туристических направлений Азии для обработки запросов
├── intents.json            # Интенты: вопросы, паттерны и ответы
├── chat_log.txt            # Лог сообщений чата
├── unknown_questions.txt   # Лог неизвестных вопросов
├── token_bot.env           # Файл с токеном бота (не хранить в публичном репо!)
├── data.pth                # Обученная модель
├── loss_plot.png           # График динамики loss при обучении
└── README.md               # Этот файл
```

---

## Установка зависимостей

Рекомендуется создать виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

Установка зависимостей:

```bash
pip install torch torchvision torchaudio
pip install python-telegram-bot==20.0
pip install nltk
pip install pymorphy2
pip install natasha
pip install matplotlib
pip install python-dotenv
```

---

## Настройка токена

Создай файл `.env` (назови `token_bot.env`) с содержимым:

```
BOT_TOKEN=твой_токен_бота
```

В коде Telegram-бота используется библиотека `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv("token_bot.env")
BOT_TOKEN = os.getenv("BOT_TOKEN")
```

---

## Обучение модели

Запуск скрипта обучения:

```bash
python train.py
```

После завершения:

- Модель сохраняется в `data.pth`
- Строится график `loss_plot.png`

---

## Параметры обучения модели

Модель чат-бота обучалась на данных из `intents.json` со следующими параметрами:

- **Размер батча (Batch size):** 16  
- **Скрытые слои (Hidden layers):** [128, 64]  
- **Learning rate:** 0.0005  
- **Количество эпох (Num epochs):** 1000  
- **Функция потерь:** CrossEntropyLoss  
- **Оптимизатор:** Adam  
- **Устройство (Device):** CPU (`Config.DEVICE`)  
- **Файл сохранённой модели:** `data.pth`  
- **График динамики loss:** `loss_plot.png`  

> В процессе обучения модель сохраняла loss по каждой эпохе, что позволяет визуализировать динамику сходимости модели.
---

## Запуск бота

```bash
python telegram_bot.py
```

Бот начинает слушать сообщения и отвечать на вопросы в Telegram.

---

## Примечания

- Все интенты хранятся в `intents.json`. Можно добавлять новые вопросы и ответы.
- Список городов для распознавания в `cities.json`.
- Для новых стран или городов добавляйте соответствующие записи в `visa_map` внутри `telegram_bot.py`.
- Логи чата ведутся в `chat_log.txt`.
- Неизвестные вопросы записываются в `unknown_questions.txt` для последующей доработки модели.
- **Не публикуйте токен бота в публичные репозитории!**

---

## Автор

Алексей Козак  
Telegram: [@TravelMateXBot](https://t.me/TravelMateXBot)