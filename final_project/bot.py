"""
ГЛАВНЫЙ МОДУЛЬ TELEGRAM БОТА TIMECOLOR
========================================
Основные функции:
1. Обработка команд пользователей (/start, /stop, /help)
2. Управление состояниями пользователей
3. Обработка и стилизация фотографий
4. Сбор статистики и обратной связи
5. Админ-панель для мониторинга

Архитектура:
- Aiogram для работы с Telegram API
- Асинхронная обработка сообщений
- Состояния пользователей для навигации по меню
- Логирование всех действий в файлы
"""

import logging
import os
import io
import numpy as np
from PIL import Image
from collections import Counter
from datetime import datetime, timedelta

# =============== ИМПОРТ БИБЛИОТЕК ===============
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

# Загружаем переменные окружения из .env файла
from dotenv import load_dotenv
load_dotenv("config.env")  # Загружает переменные из .env в окружение

# Проверяем, что файл загружен
if not os.path.exists("config.env"):
    print("⚠️ Внимание: Файл config.env не найден!")
    print("Создайте его на основе config.env.example")
    exit(1)

# Импортируем модули проекта
from utils.pipeline import process, STYLE_MAP
from utils.feedback_analyzer import feedback_analyzer

# =============== КОНФИГУРАЦИЯ ===============
"""
ВАЖНО: Все секретные данные теперь в .env файле!
.env не коммитится в git (указан в .gitignore)
"""
API_TOKEN = os.getenv("API_TOKEN")           # Токен бота из .env
ADMIN_ID = int(os.getenv("ADMIN_ID"))        # ID администратора из .env
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback.txt")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
PHOTO_STATS_FILE = os.getenv("PHOTO_STATS_FILE", "photo_stats.txt")

# Проверяем, что токен загружен
if not API_TOKEN:
    raise ValueError("❌ API_TOKEN не найден! Создайте файл .env с вашим токеном")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# =============== СОСТОЯНИЯ ПОЛЬЗОВАТЕЛЕЙ ===============
"""
Система состояний для навигации по меню:
- None: главное меню
- "photo": выбор стиля для фото
- "feedback_menu": меню обратной связи
- "feedback_typing": пользователь пишет отзыв
- "admin": админ-панель

Храним в словарях для быстрого доступа
"""
user_mode = {}      # user_id -> текущий режим (None/photo/feedback_menu и т.д.)
user_style = {}     # user_id -> выбранный стиль обработки

# =============== УТИЛИТЫ (ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ) ===============

def log_action(text: str):
    """
    Логирование действий в файл
    Используется для отладки и анализа работы бота
    Формат: [2023-12-19 14:30:45] Действие пользователя
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {text}\n")

def save_feedback(user_id: int, text: str):
    """
    Сохранение отзыва пользователя
    Формат в файле: timestamp||user_id||text
    Разделитель || выбран как редко используемый в тексте
    """
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}||{user_id}||{text}\n")
    log_action(f"Feedback from {user_id}: {text}")

def log_photo_processed(user_id: int, style: str):
    """
    Логирование обработки фото для статистики
    Формат: timestamp|user_id|style
    Позволяет анализировать популярность стилей и активность
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PHOTO_STATS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp}|{user_id}|{style}\n")
    log_action(f"Photo processed: user={user_id}, style={style}")

def get_photo_stats(days: int = None):
    """
    Получение статистики обработки фото за определенный период
    Args:
        days: количество дней для фильтрации (None - за все время)
    Returns:
        Counter: статистика по стилям
        int: общее количество фото
    """
    if not os.path.exists(PHOTO_STATS_FILE):
        return Counter(), 0
    
    with open(PHOTO_STATS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return Counter(), 0
    
    # Фильтрация по дате если указан период
    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_lines = []
        for line in lines:
            try:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    timestamp_str = parts[0]
                    line_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    if line_date >= cutoff_date:
                        filtered_lines.append(parts[2])  # стиль
            except:
                continue
        styles = filtered_lines
        total = len(filtered_lines)
    else:
        # Все время - берем все записи
        styles = [line.strip().split("|")[2] for line in lines if len(line.strip().split("|")) >= 3]
        total = len(styles)
    
    return Counter(styles), total

def create_stats_text(stats_counter, total_photos, period_text="за все время"):
    """
    Создание текстового представления статистики
    Включает:
    - Общее количество фото
    - Топ стилей с процентами
    - Текстовую диаграмму для наглядности
    """
    if total_photos == 0:
        return f"📊 Статистика обработки фото {period_text}:\n\nНет данных\n"
    
    # Сортировка стилей по популярности (по убыванию)
    sorted_stats = sorted(stats_counter.items(), key=lambda x: x[1], reverse=True)
    
    result = f"📊 Статистика обработки фото {period_text}:\n\n"
    result += f"📸 Всего обработано: {total_photos} фото\n\n"
    result += "🏆 Топ стилей:\n"
    
    # Максимальное значение для масштабирования диаграммы
    max_count = max(stats_counter.values()) if stats_counter else 1
    
    for i, (style, count) in enumerate(sorted_stats, 1):
        percentage = (count / total_photos) * 100
        # Создаем текстовую диаграмму (20 символов максимум)
        bar_length = int((count / max_count) * 20)
        bar = "█" * bar_length + " " * (20 - bar_length)
        
        result += f"{i}. {style}: {count} фото ({percentage:.1f}%)\n"
        result += f"   [{bar}] {count}/{total_photos}\n"
    
    return result

def get_detailed_stats():
    """
    Полная детальная статистика с разбивкой по периодам:
    - Сегодня
    - За 7 дней
    - За 30 дней
    - Все время
    - Активные часы
    - Статистика пользователей
    """
    if not os.path.exists(PHOTO_STATS_FILE):
        return "📊 Детальная статистика:\n\nНет данных\n"
    
    with open(PHOTO_STATS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return "📊 Детальная статистика:\n\nНет данных\n"
    
    # Рассчитываем даты для фильтрации
    today = datetime.now().date()
    week_ago = datetime.now() - timedelta(days=7)
    month_ago = datetime.now() - timedelta(days=30)
    
    # Инициализация счетчиков
    today_count = week_count = month_count = 0
    all_time_counter = Counter()
    today_counter = Counter()
    week_counter = Counter()
    month_counter = Counter()
    
    # Обработка каждой строки лога
    for line in lines:
        try:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                timestamp_str = parts[0]
                style = parts[2]
                line_datetime = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                line_date = line_datetime.date()
                
                # Все время
                all_time_counter[style] += 1
                
                # За месяц
                if line_datetime >= month_ago:
                    month_counter[style] += 1
                    month_count += 1
                    
                    # За неделю
                    if line_datetime >= week_ago:
                        week_counter[style] += 1
                        week_count += 1
                        
                        # За сегодня
                        if line_date == today:
                            today_counter[style] += 1
                            today_count += 1
        except Exception as e:
            # Пропускаем некорректные строки
            continue
    
    # Формирование результата
    result = "📊 Детальная статистика обработки фото:\n\n"
    result += f"📅 Сегодня: {today_count} фото\n"
    result += f"📅 За 7 дней: {week_count} фото\n"
    result += f"📅 За 30 дней: {month_count} фото\n"
    result += f"📅 Всего: {sum(all_time_counter.values())} фото\n\n"
    
    # Топ-5 стилей за все время
    if all_time_counter:
        result += "🏆 Топ-5 стилей за все время:\n"
        top_styles = all_time_counter.most_common(5)
        for i, (style, count) in enumerate(top_styles, 1):
            percentage = (count / sum(all_time_counter.values())) * 100
            result += f"{i}. {style}: {count} фото ({percentage:.1f}%)\n"
    
    # Самый популярный стиль сегодня
    if today_counter:
        most_popular_today = today_counter.most_common(1)[0]
        result += f"\n🔥 Самый популярный сегодня: {most_popular_today[0]} ({most_popular_today[1]} фото)\n"
    
    # Анализ активности по часам (последние 100 фото)
    hourly_stats = Counter()
    for line in lines[-100:]:  # Берем только последние 100 записей для скорости
        try:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                timestamp_str = parts[0]
                hour = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").hour
                hourly_stats[hour] += 1
        except:
            continue
    
    if hourly_stats:
        result += "\n⏰ Активные часы (последние 100 фото):\n"
        for hour in sorted(hourly_stats.keys()):
            result += f"  {hour:02d}:00 - {hourly_stats[hour]} фото\n"
    
    # Статистика пользователей
    user_stats = get_user_stats()
    result += "\n👥 Статистика пользователей:\n"
    result += f"  • Всего уникальных: {user_stats['total_users']}\n"
    result += f"  • Фото на пользователя: {user_stats['avg_photos_per_user']:.1f}\n"
    if user_stats['top_users']:
        result += f"  • Самый активный: ID {user_stats['top_users'][0][0]} "
        result += f"({user_stats['top_users'][0][1]} фото)\n"
        
    return result

def get_user_stats():
    """
    Сбор статистики по пользователям:
    - Количество уникальных пользователей
    - Активность по периодам
    - Топ пользователей по количеству фото
    - Новые пользователи
    """
    if not os.path.exists(PHOTO_STATS_FILE):
        return {
            "total_users": 0,
            "total_photos": 0,
            "photos_per_user": {},
            "active_today": 0,
            "active_week": 0,
            "active_month": 0,
            "top_users": [],
            "new_users_today": 0
        }
    
    with open(PHOTO_STATS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        return {
            "total_users": 0,
            "total_photos": 0,
            "photos_per_user": {},
            "active_today": 0,
            "active_week": 0,
            "active_month": 0,
            "top_users": [],
            "new_users_today": 0
        }
    
    # Инициализация структур данных
    user_photos = Counter()  # user_id -> количество фото
    user_first_seen = {}     # user_id -> первое появление
    user_last_seen = {}      # user_id -> последнее появление
    
    # Периоды для анализа
    today = datetime.now().date()
    week_ago = datetime.now() - timedelta(days=7)
    month_ago = datetime.now() - timedelta(days=30)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Обработка логов
    for line in lines:
        try:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                timestamp_str = parts[0]
                user_id = parts[1]
                
                line_datetime = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                
                # Считаем фото по пользователям
                user_photos[user_id] += 1
                
                # Запоминаем первое и последнее появление
                if user_id not in user_first_seen or line_datetime < user_first_seen[user_id]:
                    user_first_seen[user_id] = line_datetime
                
                if user_id not in user_last_seen or line_datetime > user_last_seen[user_id]:
                    user_last_seen[user_id] = line_datetime
                    
        except Exception as e:
            continue
    
    # Подсчет активных пользователей
    active_today = active_week = active_month = new_users_today = 0
    
    for user_id, last_seen in user_last_seen.items():
        if last_seen >= today_start:
            active_today += 1
        
        if last_seen >= week_ago:
            active_week += 1
        
        if last_seen >= month_ago:
            active_month += 1
    
    # Новые пользователи за сегодня
    for user_id, first_seen in user_first_seen.items():
        if first_seen >= today_start:
            new_users_today += 1
    
    # Топ пользователей по количеству фото
    top_users = user_photos.most_common(10)
    
    return {
        "total_users": len(user_photos),
        "total_photos": sum(user_photos.values()),
        "photos_per_user": dict(user_photos),
        "active_today": active_today,
        "active_week": active_week,
        "active_month": active_month,
        "top_users": top_users,
        "new_users_today": new_users_today,
        "avg_photos_per_user": sum(user_photos.values()) / len(user_photos) if user_photos else 0
    }

def create_user_stats_text():
    """Создание текстового представления статистики пользователей"""
    stats = get_user_stats()
    
    if stats["total_users"] == 0:
        return "📊 Статистика пользователей:\n\nНет данных\n"
    
    result = "👥 Статистика пользователей:\n\n"
    
    result += f"👤 Всего пользователей: {stats['total_users']}\n"
    result += f"📸 Всего обработано фото: {stats['total_photos']}\n"
    result += f"📊 Среднее фото на пользователя: {stats['avg_photos_per_user']:.1f}\n\n"
    
    result += "📅 Активность пользователей:\n"
    result += f"  • Сегодня: {stats['active_today']} пользователей\n"
    result += f"  • За 7 дней: {stats['active_week']} пользователей\n"
    result += f"  • За 30 дней: {stats['active_month']} пользователей\n"
    result += f"  • Новых сегодня: {stats['new_users_today']}\n\n"
    
    if stats["top_users"]:
        result += "🏆 Топ-10 активных пользователей:\n"
        for i, (user_id, count) in enumerate(stats["top_users"][:10], 1):
            percentage = (count / stats["total_photos"]) * 100 if stats["total_photos"] > 0 else 0
            result += f"{i}. ID {user_id}: {count} фото ({percentage:.1f}%)\n"
    
    # Распределение по активности
    photo_counts = Counter(stats["photos_per_user"].values())
    if photo_counts:
        result += "\n📈 Распределение по активности:\n"
        sorted_counts = sorted(photo_counts.items())
        for count, users in sorted_counts:
            if count <= 10:
                result += f"  • {users} пользователей сделали {count} фото\n"
        
        more_than_10 = sum(users for cnt, users in photo_counts.items() if cnt > 10)
        if more_than_10 > 0:
            result += f"  • {more_than_10} пользователей сделали более 10 фото\n"
    
    return result
    
@dp.message_handler(lambda m: m.text == "✅ Включить уведомления" and m.from_user.id == ADMIN_ID)
async def enable_all_notifications(msg: types.Message):
    """Включить ВСЕ категории уведомлений"""
    settings = feedback_analyzer.notification_settings
    
    # Включаем ВСЕ категории
    for category in settings["categories"]:
        settings["categories"][category] = True
    
    feedback_analyzer.save_settings()
    
    await msg.answer("✅ Все категории уведомлений включены!",
                    reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "🔕 Выключить уведомления" and m.from_user.id == ADMIN_ID)
async def disable_all_notifications(msg: types.Message):
    """Выключить ВСЕ категории уведомлений"""
    settings = feedback_analyzer.notification_settings
    
    # Выключаем ВСЕ категории
    for category in settings["categories"]:
        settings["categories"][category] = False
    
    feedback_analyzer.save_settings()
    
    await msg.answer("🔕 Все категории уведомлений выключены!",
                    reply_markup=notifications_menu())
                    
async def send_admin_notification(bot, user_id, feedback_text, analysis):
    """Отправляет уведомление админу о критичном отзыве"""
    try:
        # Проверяем только категорию (общего статуса больше нет)
        category = analysis["main_category"]
        
        # Получаем настройки категории напрямую
        category_settings = feedback_analyzer.notification_settings.get("categories", {})
        
        if not category_settings.get(category, False):
            log_action(f"Уведомление НЕ отправлено: категория {category} выключена")
            return
        
        # Форматируем сообщение
        notification = feedback_analyzer.format_notification_message(
            user_id, feedback_text, analysis
        )
        
        # Отправляем
        await bot.send_message(
            chat_id=ADMIN_ID,
            text=notification,
            parse_mode="Markdown"
        )
        
        log_action(f"Уведомление отправлено: {category} от пользователя {user_id}")
        
    except Exception as e:
        print(f"Ошибка при отправке уведомления: {e}")
        log_action(f"Notification error: {e}")

@dp.message_handler(commands=["stats"])
async def quick_stats(msg: types.Message):
    """
    Обработчик команды /stats
    Показывает краткую статистику только администратору
    """
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступно только администратору")
        return
    
    # Сбор статистики
    photo_stats, total_photos = get_photo_stats()
    user_stats = get_user_stats()
    
    # Формирование ответа
    stats_text = "📊 Быстрая статистика:\n\n"
    stats_text += f"👥 Пользователей: {user_stats['total_users']}\n"
    stats_text += f"📸 Обработано фото: {total_photos}\n"
    stats_text += f"📅 Активных сегодня: {user_stats['active_today']}\n\n"
    
    if photo_stats:
        top_style = photo_stats.most_common(1)[0]
        stats_text += f"🏆 Популярный стиль: {top_style[0]} ({top_style[1]} фото)\n"
    
    await msg.answer(stats_text)
    
@dp.message_handler(commands=["feedback_stats"])
async def show_feedback_stats(msg: types.Message):
    """Показать статистику отзывов"""
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступно только администратору")
        return
    
    stats_text = feedback_analyzer.get_stats_summary()
    await msg.answer(stats_text, parse_mode="Markdown")

@dp.message_handler(commands=["feedback_insights"])
async def show_feedback_insights(msg: types.Message):
    """Показать инсайты из отзывов"""
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступно только администратору")
        return
    
    insights = feedback_analyzer.get_insights()
    
    text = "🔍 *Инсайты из отзывов:*\n\n"
    text += insights
    
    await msg.answer(text, parse_mode="Markdown")

@dp.message_handler(commands=["latest_feedback"])
async def show_latest_feedback(msg: types.Message):
    """Показать последние отзывы с анализом"""
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступно только администратору")
        return
    
    if not os.path.exists(FEEDBACK_FILE):
        await msg.answer("Отзывов пока нет.")
        return
    
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-10:]  # Последние 10 отзывов
    
    if not lines:
        await msg.answer("Отзывов пока нет.")
        return
    
    text = "📝 *Последние отзывы:*\n\n"
    
    for i, line in enumerate(lines, 1):
        if "||" in line:
            parts = line.strip().split("||")
            if len(parts) >= 3:
                timestamp = parts[0][:16]  # Берем дату и время
                user_id = parts[1]
                feedback = parts[2]
                
                # Анализируем отзыв
                analysis = feedback_analyzer.analyze_feedback(feedback)
                
                # Обрезаем длинный текст
                short_feedback = feedback[:50] + "..." if len(feedback) > 50 else feedback
                
                text += f"{analysis['emoji']} *{analysis['main_category'].upper()}* "
                text += f"(уверенность: {analysis['confidence']*100:.0f}%)\n"
                text += f"👤 User {user_id} | 📅 {timestamp}\n"
                text += f"💬 \"{short_feedback}\"\n\n"
    
    await msg.answer(text, parse_mode="Markdown")

def read_feedback(last_n=50):
    """
    Чтение отзывов из файла
    Args:
        last_n: количество последних отзывов для чтения
    Returns:
        list: отформатированные строки отзывов
    """
    if not os.path.exists(FEEDBACK_FILE):
        return []
    
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Форматирование для красивого вывода
    formatted_lines = []
    for line in lines[-last_n:]:
        parts = line.strip().split("||")
        if len(parts) >= 3:
            timestamp = parts[0]
            user_id = parts[1]
            text = parts[2]
            formatted_lines.append(f"[{timestamp}] User {user_id}: {text}\n")
    
    return formatted_lines

# =============== СИСТЕМА МЕНЮ ===============
"""
Все меню создаются как функции для переиспользования
ReplyKeyboardMarkup - клавиатура ответа в Telegram
resize_keyboard=True - автоматическое масштабирование
"""

def main_menu(user_id: int):
    """
    Главное меню бота
    Для администратора добавляется дополнительная кнопка
    """
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = [
        KeyboardButton("📸 Обработать фото"),
        KeyboardButton("💬 Обратная связь"),
        KeyboardButton("❓ Помощь")
    ]
    
    # Добавляем кнопку админ-панели только для администратора
    if user_id == ADMIN_ID:
        buttons.append(KeyboardButton("👑 Админ-панель"))
    
    kb.add(*buttons)
    return kb

def styles_menu():
    """
    Меню выбора стиля обработки
    row_width=2 - 2 кнопки в строке для компактности
    """
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        "Реалистично 🖼️",
        "Холодные тона ❄️",
        "Ретро / Сепия 🎞️",
        "Советские 📻",
        "Дореволюционные 🏰",
        "90-е 📼",
        "Современные 📷",
        "Винтаж Европа/Америка 🇪🇺",
        "Ретро (50-80) 🕰️",
        "⬅️ Назад"
    ]
    kb.add(*[KeyboardButton(b) for b in buttons])
    return kb

def feedback_menu():
    """Меню обратной связи"""
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(
        KeyboardButton("✏️ Отправить отзыв"),
        KeyboardButton("📄 Посмотреть мои отзывы"),
        KeyboardButton("⬅️ Назад")
    )
    return kb

def admin_menu():
    """Админ-панель с опциями статистики и управления"""
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(
        KeyboardButton("📊 Статистика фото"),
        KeyboardButton("👥 Статистика пользователей"),
        KeyboardButton("📈 Детальная статистика"),
        KeyboardButton("📝 Анализ отзывов"), 
        KeyboardButton("📄 Все отзывы"),
        KeyboardButton("🔔 Управление уведомлениями"), 
        KeyboardButton("⬅️ Назад")
    )
    return kb
    
def notifications_menu():
    """Меню управления уведомлениями"""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add(
        KeyboardButton("🔔 Статус уведомлений"),
        KeyboardButton("✅ Включить уведомления"),
        KeyboardButton("🔕 Выключить уведомления"),
        KeyboardButton("📋 Категории уведомлений"),
        KeyboardButton("📝 Тестовое уведомление"),
        KeyboardButton("⬅️ Назад в админ")
    )
    return kb
    
def categories_menu():
    """Меню управления категориями уведомлений (БЕЗ кнопок включения/выключения)"""
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    
    # Только кнопки категорий
    categories = [
        "🐛 Баги (bug)",
        "❌ Жалобы (complaint)", 
        "✅ Похвалы (praise)",
        "💡 Предложения (suggestion)",
        "❓ Вопросы (question)"
    ]
    
    # Добавляем каждую категорию отдельной кнопкой
    for category in categories:
        kb.add(KeyboardButton(category))
    
    # УБИРАЕМ "Включить все категории" и "Выключить все категории"
    # Оставляем только кнопку назад
    kb.add(KeyboardButton("⬅️ Назад в уведомления"))
    return kb

def stats_period_menu():
    """Меню выбора периода для статистики"""
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add(
        KeyboardButton("📊 Статистика за сегодня"),
        KeyboardButton("📊 За 7 дней"),
        KeyboardButton("📊 За 30 дней"),
        KeyboardButton("📊 За все время"),
        KeyboardButton("⬅️ Назад в админ")
    )
    return kb

# =============== ОБРАБОТЧИКИ КОМАНД ===============
"""
Декораторы @dp.message_handler определяют, какие сообщения обрабатывать
lambda функции - фильтры для определенного текста или условий
"""

@dp.message_handler(commands=["start"])
async def start(msg: types.Message):
    """
    Обработчик команды /start
    Сбрасывает состояние пользователя и показывает главное меню
    """
    user_mode[msg.from_user.id] = None  # Сброс в главное меню
    await msg.answer("Главное меню:", reply_markup=main_menu(msg.from_user.id))
    
@dp.message_handler(commands=["stop"])
async def stop_user_session(msg: types.Message):
    """
    Обработчик команды /stop
    Полностью сбрасывает сессию пользователя:
    - Удаляет состояние
    - Удаляет выбранный стиль
    - Показывает клавиатуру удаления
    """
    user_id = msg.from_user.id
    
    # Очистка данных пользователя
    if user_id in user_mode:
        del user_mode[user_id]
    
    if user_id in user_style:
        del user_style[user_id]
    
    log_action(f"User {user_id} stopped session")
    
    await msg.answer(
        "👋 Сессия завершена!\n"
        "Все ваши данные сброшены.\n"
        "Для начала работы нажмите /start",
        reply_markup=types.ReplyKeyboardRemove()  # Убираем клавиатуру
    )

# ---------- АДМИН ПАНЕЛЬ ----------
@dp.message_handler(lambda m: m.text == "👑 Админ-панель")
async def admin_panel(msg: types.Message):
    """
    Вход в админ-панель
    Проверка по ID администратора
    """
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступ запрещен")
        return
    user_mode[msg.from_user.id] = "admin"
    await msg.answer("Админ-панель:", reply_markup=admin_menu())
    
@dp.message_handler(lambda m: m.text == "📝 Анализ отзывов" and m.from_user.id == ADMIN_ID)
async def show_feedback_analysis(msg: types.Message):
    # Показываем меню анализа
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add(
        KeyboardButton("📊 Статистика отзывов"),
        KeyboardButton("🔍 Инсайты"),
        KeyboardButton("📋 Последние отзывы"),
        KeyboardButton("⬅️ Назад в админ")
    )
    
    await msg.answer("Выберите тип анализа отзывов:", reply_markup=kb)

@dp.message_handler(lambda m: m.text == "📊 Статистика отзывов" and m.from_user.id == ADMIN_ID)
async def show_feedback_stats_btn(msg: types.Message):
    await show_feedback_stats(msg)

@dp.message_handler(lambda m: m.text == "🔍 Инсайты" and m.from_user.id == ADMIN_ID)
async def show_feedback_insights_btn(msg: types.Message):
    await show_feedback_insights(msg)

@dp.message_handler(lambda m: m.text == "📋 Последние отзывы" and m.from_user.id == ADMIN_ID)
async def show_latest_feedback_btn(msg: types.Message):
    await show_latest_feedback(msg)

# ---------- УПРАВЛЕНИЕ УВЕДОМЛЕНИЯМИ ----------
@dp.message_handler(commands=["test_notify"])
async def test_notification_cmd(msg: types.Message):
    """Тест уведомления"""
    if msg.from_user.id != ADMIN_ID:
        return
    
    # Тестовые данные
    test_feedback = "ТЕСТ: Какой максимальный размер фото поддерживается?"
    test_analysis = feedback_analyzer.analyze_feedback(test_feedback, msg.from_user.id)
    
    # Показываем анализ
    debug_info = f"""
🔍 *ТЕСТ УВЕДОМЛЕНИЯ*

📝 Текст: {test_feedback}
🏷️ Категория: {test_analysis['main_category']} ({test_analysis['emoji']})
📊 Уверенность: {test_analysis['confidence']*100:.0f}%

⚙️ *ПРОВЕРКИ:*
2. Категория '{test_analysis['main_category']}' включена: {feedback_analyzer.is_category_enabled(test_analysis['main_category'])}
3. Кулдаун пройден: {feedback_analyzer.should_send_notification(msg.from_user.id, test_feedback)}
    """
    
    await msg.answer(debug_info, parse_mode="Markdown")
    
    # Пробуем отправить
    await send_admin_notification(bot, msg.from_user.id, test_feedback, test_analysis)
    
    await msg.answer("✅ Тест завершен. Проверьте, пришло ли уведомление.")
    
@dp.message_handler(lambda m: m.text == "🔔 Управление уведомлениями" and m.from_user.id == ADMIN_ID)
async def show_notifications_menu(msg: types.Message):
    """Показ меню управления уведомлениями"""
    await msg.answer("Управление уведомлениями:", reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "🔔 Статус уведомлений" and m.from_user.id == ADMIN_ID)
async def show_notification_status(msg: types.Message):
    """Показать статус уведомлений"""
    settings = feedback_analyzer.notification_settings
    categories = settings.get("categories", {})
    
    text = "🔔 *СТАТУС УВЕДОМЛЕНИЙ*\n\n"
    
    # Проверяем состояние
    all_enabled = all(categories.values()) if categories else False
    all_disabled = not any(categories.values()) if categories else True
    
    if all_enabled:
        text += "📢 Все уведомления: ✅ ВКЛЮЧЕНЫ\n\n"
    elif all_disabled:
        text += "📢 Все уведомления: 🔕 ВЫКЛЮЧЕНЫ\n\n"
    else:
        text += "📢 Все уведомления: ⚙️ ЧАСТИЧНО ВКЛЮЧЕНЫ\n\n"
    
    text += "📋 Состояние категорий:\n"
    
    category_names = {
        "bug": "🐛 Баги",
        "complaint": "❌ Жалобы", 
        "praise": "✅ Похвалы",
        "suggestion": "💡 Предложения",
        "question": "❓ Вопросы"
    }
    
    # Показываем состояние каждой категории
    enabled_count = 0
    for category_id, name in category_names.items():
        enabled = categories.get(category_id, False)
        status = "✅ ВКЛ" if enabled else "🔕 ВЫКЛ"
        text += f"{status} {name}\n"
        if enabled:
            enabled_count += 1
    
    total_categories = len(categories)
    text += f"\n📊 Включено категорий: {enabled_count}/{total_categories}"
    
    if enabled_count == 0:
        text += "\n⚠️ Никакие уведомления не будут приходить"
    elif enabled_count == total_categories:
        text += "\n✅ Будут приходить все уведомления"
    else:
        text += f"\n⚙️ Будут приходить только выбранные уведомления"
    
    text += "\n\nДля изменения используйте меню ниже 👇"
    
    await msg.answer(text, parse_mode="Markdown", reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "✅ Включить уведомления" and m.from_user.id == ADMIN_ID)
async def enable_notifications_btn(msg: types.Message):
    """Включить общие уведомления (БЕЗ автоматического включения категорий)"""
    settings = feedback_analyzer.notification_settings
    
    # Включаем только общий флаг
    #settings["enabled"] = True
    
    # НЕ включаем категории автоматически!
    # Они остаются такими, какими были настроены ранее
    
    feedback_analyzer.save_settings()
    
    await msg.answer("✅ Общие уведомления включены!\n"
                    "Теперь включите нужные категории отдельно.",
                    reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "🔕 Выключить уведомления" and m.from_user.id == ADMIN_ID)
async def disable_notifications_btn(msg: types.Message):
    """Выключить все уведомления через кнопку в основном меню"""
    settings = feedback_analyzer.notification_settings
    
    # 1. Выключаем общий флаг
    #settings["enabled"] = False
    
    # 2. Выключать категории НЕ нужно - они будут игнорироваться если общий флаг выключен
    # Но можно оставить как есть или выключить их тоже
    # for category in settings["categories"]:
    #     settings["categories"][category] = False
    
    feedback_analyzer.save_settings()
    
    await msg.answer("🔕 Все уведомления выключены!\n"
                    "Это отключает ВСЕ уведомления, независимо от настроек категорий.",
                    reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "📋 Категории уведомлений" and m.from_user.id == ADMIN_ID)
async def manage_categories(msg: types.Message):
    """Управление категориями уведомлений"""
    settings = feedback_analyzer.notification_settings
    categories = settings.get("categories", {})
    
    text = "📋 *УПРАВЛЕНИЕ КАТЕГОРИЯМИ УВЕДОМЛЕНИЙ*\n\n"
    
    # Показываем общую статистику
    enabled_count = sum(1 for enabled in categories.values() if enabled)
    total_categories = len(categories)
    
    if enabled_count == 0:
        text += "📢 Статус: 🔕 ВСЕ ВЫКЛЮЧЕНЫ\n\n"
    elif enabled_count == total_categories:
        text += "📢 Статус: ✅ ВСЕ ВКЛЮЧЕНЫ\n\n"
    else:
        text += f"📢 Статус: ⚙️ {enabled_count}/{total_categories} включено\n\n"
    
    text += "Текущее состояние:\n"
    
    category_status = {
        "bug": "🐛 Баги",
        "complaint": "❌ Жалобы", 
        "praise": "✅ Похвалы",
        "suggestion": "💡 Предложения",
        "question": "❓ Вопросы"
    }
    
    # Показываем реальный статус категорий
    for category_id, name in category_status.items():
        enabled = categories.get(category_id, False)
        status = "✅ ВКЛ" if enabled else "🔕 ВЫКЛ"
        text += f"{status} {name}\n"
    
    text += "\nНажмите на категорию, чтобы переключить её состояние 👇"
    
    await msg.answer(text, parse_mode="Markdown", reply_markup=categories_menu())

async def send_test_notification_force(bot, user_id):
    """Принудительно отправить тестовое уведомление"""
    try:
        test_feedback = "🐛 ТЕСТ: Проверка работы уведомлений от админ-панели"
        
        # Создаем принудительный анализ как BUG
        test_analysis = {
            "main_category": "bug",
            "confidence": 1.0,
            "emoji": "🐛"
        }
        
        # Форматируем сообщение напрямую
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        
        notification = f"""
🐛 *ТЕСТОВОЕ УВЕДОМЛЕНИЕ*

👤 Пользователь: `{user_id}`
📊 Категория: BUG (ТЕСТ)
🕒 Время: {current_time}
📈 Уверенность: 100%

💬 Текст:
{test_feedback}
        """
        
        # Отправляем напрямую, минуя все проверки
        await bot.send_message(
            chat_id=ADMIN_ID,
            text=notification,
            parse_mode="Markdown"
        )
        
        return True
    except Exception as e:
        print(f"Ошибка отправки тестового уведомления: {e}")
        return False

@dp.message_handler(lambda m: m.text == "📝 Тестовое уведомление" and m.from_user.id == ADMIN_ID)
async def send_test_notification(msg: types.Message):
    """Отправить тестовое уведомление"""
    try:
        success = await send_test_notification_force(bot, msg.from_user.id)
        if success:
            await msg.answer("✅ Тестовое уведомление отправлено! Проверьте личные сообщения от бота.", 
                            reply_markup=notifications_menu())
        else:
            await msg.answer("❌ Ошибка отправки тестового уведомления.", 
                            reply_markup=notifications_menu())
    except Exception as e:
        await msg.answer(f"❌ Ошибка: {str(e)}", reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "🐛 Баги (bug)" and m.from_user.id == ADMIN_ID)
async def toggle_bug_notifications(msg: types.Message):
    """Переключить уведомления о багах"""
    settings = feedback_analyzer.notification_settings
    
    # УБЕРИТЕ эту проверку - разрешаем менять всегда
    # if not settings["enabled"]:
    #     await msg.answer("⚠️ Сначала включите уведомления в основном меню!", 
    #                     reply_markup=notifications_menu())
    #     return
    
    current = settings["categories"]["bug"]
    settings["categories"]["bug"] = not current
    feedback_analyzer.save_settings()
    
    status = "✅ ВКЛЮЧЕНЫ" if not current else "🔕 ВЫКЛЮЧЕНЫ"
    await msg.answer(f"Уведомления о багах {status}", reply_markup=categories_menu())

@dp.message_handler(lambda m: m.text == "❌ Жалобы (complaint)" and m.from_user.id == ADMIN_ID)
async def toggle_complaint_notifications(msg: types.Message):
    """Переключить уведомления о жалобах"""
    settings = feedback_analyzer.notification_settings

    current = feedback_analyzer.notification_settings["categories"]["complaint"]
    feedback_analyzer.notification_settings["categories"]["complaint"] = not current
    feedback_analyzer.save_settings()
    
    status = "✅ ВКЛЮЧЕНЫ" if not current else "🔕 ВЫКЛЮЧЕНЫ"
    await msg.answer(f"Уведомления о жалобах {status}", reply_markup=categories_menu())

@dp.message_handler(lambda m: m.text == "✅ Похвалы (praise)" and m.from_user.id == ADMIN_ID)
async def toggle_praise_notifications(msg: types.Message):
    """Переключить уведомления о похвалах"""
    settings = feedback_analyzer.notification_settings
    
    current = feedback_analyzer.notification_settings["categories"]["praise"]
    feedback_analyzer.notification_settings["categories"]["praise"] = not current
    feedback_analyzer.save_settings()
    
    status = "✅ ВКЛЮЧЕНЫ" if not current else "🔕 ВЫКЛЮЧЕНЫ"
    await msg.answer(f"Уведомления о похвалах {status}", reply_markup=categories_menu())

@dp.message_handler(lambda m: m.text == "💡 Предложения (suggestion)" and m.from_user.id == ADMIN_ID)
async def toggle_suggestion_notifications(msg: types.Message):
    """Переключить уведомления о предложениях"""
    settings = feedback_analyzer.notification_settings

    current = feedback_analyzer.notification_settings["categories"]["suggestion"]
    feedback_analyzer.notification_settings["categories"]["suggestion"] = not current
    feedback_analyzer.save_settings()
    
    status = "✅ ВКЛЮЧЕНЫ" if not current else "🔕 ВЫКЛЮЧЕНЫ"
    await msg.answer(f"Уведомления о предложениях {status}", reply_markup=categories_menu())

@dp.message_handler(lambda m: m.text == "❓ Вопросы (question)" and m.from_user.id == ADMIN_ID)
async def toggle_question_notifications(msg: types.Message):
    """Переключить уведомления о вопросах"""
    settings = feedback_analyzer.notification_settings

    current = feedback_analyzer.notification_settings["categories"]["question"]
    feedback_analyzer.notification_settings["categories"]["question"] = not current
    feedback_analyzer.save_settings()
    
    status = "✅ ВКЛЮЧЕНЫ" if not current else "🔕 ВЫКЛЮЧЕНЫ"
    await msg.answer(f"Уведомления о вопросах {status}", reply_markup=categories_menu())
'''
@dp.message_handler(lambda m: m.text == "✅ Включить все категории" and m.from_user.id == ADMIN_ID)
async def enable_all_categories(msg: types.Message):
    """Включить все категории уведомлений"""
    for category in feedback_analyzer.notification_settings["categories"]:
        feedback_analyzer.notification_settings["categories"][category] = True
    feedback_analyzer.save_settings()
    
    await msg.answer("✅ Все категории уведомлений включены!", reply_markup=categories_menu())

@dp.message_handler(lambda m: m.text == "🔕 Выключить все категории" and m.from_user.id == ADMIN_ID)
async def disable_all_categories(msg: types.Message):
    """Выключить все категории уведомлений"""
    for category in feedback_analyzer.notification_settings["categories"]:
        feedback_analyzer.notification_settings["categories"][category] = False
    feedback_analyzer.save_settings()
    
    await msg.answer("🔕 Все категории уведомлений выключены!", reply_markup=categories_menu())
'''
# ---------- КНОПКИ ВОЗВРАТА ----------
@dp.message_handler(lambda m: m.text == "⬅️ Назад в уведомления" and m.from_user.id == ADMIN_ID)
async def back_to_notifications(msg: types.Message):
    """Вернуться в меню уведомлений"""
    await msg.answer("Управление уведомлениями:", reply_markup=notifications_menu())

@dp.message_handler(lambda m: m.text == "⬅️ Назад в админ" and m.from_user.id == ADMIN_ID)
async def back_to_admin_from_notifications(msg: types.Message):
    """Вернуться в админ-панель из меню уведомлений"""
    user_mode[msg.from_user.id] = "admin"
    await msg.answer("Админ-панель:", reply_markup=admin_menu())

# ---------- СТАТИСТИКА ----------
@dp.message_handler(lambda m: m.text == "📊 Статистика фото" and m.from_user.id == ADMIN_ID)
async def show_stats_options(msg: types.Message):
    """Показ меню выбора периода статистики"""
    await msg.answer("Выберите период для статистики:", reply_markup=stats_period_menu())

@dp.message_handler(lambda m: m.text == "📊 Статистика за сегодня" and m.from_user.id == ADMIN_ID)
async def stats_today(msg: types.Message):
    """Статистика за сегодняшний день"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    stats_counter, total = get_photo_stats()
    
    # Фильтрация вручную для точности
    if os.path.exists(PHOTO_STATS_FILE):
        with open(PHOTO_STATS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        today_counter = Counter()
        today_total = 0
        
        for line in lines:
            try:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    timestamp_str = parts[0]
                    style = parts[2]
                    line_datetime = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    if line_datetime >= today_start:
                        today_counter[style] += 1
                        today_total += 1
            except:
                continue
        
        stats_text = create_stats_text(today_counter, today_total, "за сегодня")
    else:
        stats_text = "📊 Статистика за сегодня:\n\nНет данных\n"
    
    await msg.answer(stats_text)

@dp.message_handler(lambda m: m.text == "📊 За 7 дней" and m.from_user.id == ADMIN_ID)
async def stats_week(msg: types.Message):
    """Статистика за последние 7 дней"""
    stats_counter, total = get_photo_stats(days=7)
    stats_text = create_stats_text(stats_counter, total, "за 7 дней")
    await msg.answer(stats_text)

@dp.message_handler(lambda m: m.text == "📊 За 30 дней" and m.from_user.id == ADMIN_ID)
async def stats_month(msg: types.Message):
    """Статистика за последние 30 дней"""
    stats_counter, total = get_photo_stats(days=30)
    stats_text = create_stats_text(stats_counter, total, "за 30 дней")
    await msg.answer(stats_text)

@dp.message_handler(lambda m: m.text == "📊 За все время" and m.from_user.id == ADMIN_ID)
async def stats_all_time(msg: types.Message):
    """Статистика за все время работы бота"""
    stats_counter, total = get_photo_stats()
    stats_text = create_stats_text(stats_counter, total, "за все время")
    await msg.answer(stats_text)

@dp.message_handler(lambda m: m.text == "📈 Детальная статистика" and m.from_user.id == ADMIN_ID)
async def detailed_stats(msg: types.Message):
    """Полная детальная статистика"""
    stats_text = get_detailed_stats()
    await msg.answer(stats_text)
    
@dp.message_handler(lambda m: m.text == "👥 Статистика пользователей" and m.from_user.id == ADMIN_ID)
async def show_user_stats(msg: types.Message):
    """Статистика по пользователям"""
    stats_text = create_user_stats_text()
    await msg.answer(stats_text)

@dp.message_handler(lambda m: m.text == "⬅️ Назад в админ" and m.from_user.id == ADMIN_ID)
async def back_to_admin(msg: types.Message):
    """Возврат в админ-панель из меню статистики"""
    user_mode[msg.from_user.id] = "admin"
    await msg.answer("Админ-панель:", reply_markup=admin_menu())

# ---------- ГЛАВНОЕ МЕНЮ ----------
@dp.message_handler(lambda m: m.text == "📸 Обработать фото")
async def go_photo(msg: types.Message):
    """
    Переход в режим обработки фото
    Устанавливает состояние "photo" и показывает меню стилей
    """
    user_mode[msg.from_user.id] = "photo"
    await msg.answer("Выбери стиль обработки:", reply_markup=styles_menu())

@dp.message_handler(lambda m: m.text == "💬 Обратная связь")
async def go_feedback(msg: types.Message):
    """
    Переход в меню обратной связи
    Устанавливает состояние "feedback_menu"
    """
    user_mode[msg.from_user.id] = "feedback_menu"
    await msg.answer("Меню обратной связи:", reply_markup=feedback_menu())

# ---------- НАЗАД ----------
@dp.message_handler(lambda m: m.text == "⬅️ Назад")
async def go_back(msg: types.Message):
    """
    Возврат в главное меню
    Очищает состояние и выбранный стиль
    """
    user_mode[msg.from_user.id] = None
    user_style.pop(msg.from_user.id, None)  # Удаляем выбранный стиль
    
    # Возвращаемся в главное меню с учетом прав
    await msg.answer("Главное меню:", reply_markup=main_menu(msg.from_user.id))
    
@dp.message_handler(lambda m: m.text == "❓ Помощь")
async def show_help(msg: types.Message):
    """
    Показ справки по использованию бота
    Подробное описание функций и советов
    """
    help_text = """
🤖 *Помощь по боту TimeColor:*

*Основные команды:*
/start - Главное меню
/stop - Завершить сессию (сбросить все данные)
/help - Эта справка

*Как пользоваться:*
1. Нажмите "📸 Обработать фото"
2. Выберите стиль обработки из предложенных
3. Отправьте фото (можно сжать)
4. Получите обработанное фото!

*📸 Особенности обработки:*
• *Для черно-белых фото:*
  - Автоматически раскрашиваются перед обработкой
  - Все стили доступны с наилучшим качеством
  - Особенно рекомендуются: Реалистично, Современные

• *Для цветных фото:*
  - Стили "Реалистично" и "Современные" не применяются
  - Возвращается исходное фото (чтобы не ухудшить качество)
  - Все остальные стили работают как обычно

*🎨 Описание стилей:*
• *Реалистично* 🖼️ - естественное раскрашивание ЧБ фото
• *Холодные тона* ❄️ - холодная, синеватая цветовая гамма
• *Ретро / Сепия* 🎞️ - классический коричневый фильтр
• *Советские* 📻 - стиль старых советских фото
• *Дореволюционные* 🏰 - эффект старины с лёгким размытием
• *90-е* 📼 - яркие, насыщенные цвета в стиле 90-х
• *Современные* 📷 - современное раскрашивание ЧБ фото
• *Винтаж Европа/Америка* 🇪🇺 - винтажный эффект
• *Ретро (50-80)* 🕰️ - стиль середины 20 века с виньетированием

*💡 Советы для лучшего результата:*
1. *Черно-белые фото:*
   - Лучше всего работают чёткие, контрастные фото
   - Хорошее освещение на исходнике = лучшее качество
   - Рекомендуемые стили: Реалистично, Современные

2. *Цветные фото:*
   - Избегайте стилей "Реалистично" и "Современные"
   - Для улучшения цвета используйте "Холодные тона"
   - Для винтажного эффекта: Сепия, Винтаж, Ретро

3. *Общие рекомендации:*
   - Отправляйте фото в хорошем качестве
   - Избегайте сильно сжатых изображений
   - Для портретов лучше всего подходит "Реалистично"
   - Для пейзажей попробуйте "Холодные тона" или "90-е"

*🔄 Что делать если:*
• *Результат не понравился* - попробуйте другой стиль
• *Фото не обрабатывается* - проверьте интернет, отправьте заново
• *Хотите начать заново* - используйте /stop
• *Есть вопросы/предложения* - нажмите "💬 Обратная связь"

*❓ Частые вопросы:*
Q: Почему для цветных фото не работают "Реалистично" и "Современные"?
A: Эти стили предназначены для раскрашивания ЧБ фото. Применение их к цветным фото ухудшит качество.

Q: Какой стиль лучше для старых семейных фото?
A: "Реалистично" для естественного цвета, "Советские" или "Ретро (50-80)" для стилизации.

Q: Можно ли обрабатывать скриншоты?
A: Да, но лучше использовать стили "Холодные тона" или "Современные".

Для дополнительной помощи или отзыва используйте меню "💬 Обратная связь".
"""
    
    # Просто отправляем help - пользователь УЖЕ видит меню
    await msg.answer(help_text, parse_mode="Markdown")
    # Ничего больше не отправляем - не меняем меню

# ---------- ВЫБОР СТИЛЯ ----------
@dp.message_handler(lambda m: any(s in m.text for s in STYLE_MAP))
async def choose_style(msg: types.Message):
    """
    Обработчик выбора стиля
    Ищет название стиля в тексте сообщения
    """
    if user_mode.get(msg.from_user.id) != "photo":
        return

    # Поиск стиля в тексте сообщения
    style = next((s for s in STYLE_MAP if s in msg.text), None)
    if not style:
        await msg.answer("❌ Некорректный стиль")
        return

    # Сохраняем выбранный стиль
    user_style[msg.from_user.id] = style
    await msg.answer(f"✅ Выбран стиль: {style}\nТеперь отправь фото 📸")

# ---------- ОБРАБОТКА ФОТО ----------
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(msg: types.Message):
    """
    Обработчик загрузки фото
    Основной алгоритм:
    1. Проверка состояния пользователя
    2. Загрузка фото из Telegram
    3. Конвертация в numpy массив
    4. Обработка через pipeline
    5. Отправка результата
    """
    # Проверка состояния
    if user_mode.get(msg.from_user.id) != "photo":
        await msg.answer("❌ Сейчас нельзя отправлять фото.")
        return

    # Проверка выбранного стиля
    style = user_style.get(msg.from_user.id)
    if not style:
        await msg.answer("❗ Сначала выбери стиль.")
        return

    # Загрузка фото (Telegram отправляет несколько размеров, берем самый большой)
    photo = msg.photo[-1]
    file = await bot.download_file_by_id(photo.file_id)
    
    # Конвертация в PIL Image и numpy array
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img_np = np.array(img) / 255.0  # Нормализация [0, 1]

    await msg.answer("Обрабатываю... ⏳")

    try:
        # Основная обработка через pipeline
        out_np = process(img_np, style)
        
        # Конвертация обратно в PIL Image
        out_img = Image.fromarray((out_np * 255).astype(np.uint8))

        # Подготовка для отправки в Telegram
        bio = io.BytesIO()
        bio.name = "result.jpg"
        out_img.save(bio, "JPEG", quality=95)
        bio.seek(0)  # Возврат в начало для чтения

        # Логирование для статистики
        log_photo_processed(msg.from_user.id, style)
        
        # Отправка результата
        await bot.send_photo(msg.chat.id, bio, caption=f"Стиль: {style}")
        
    except Exception as e:
        # Обработка ошибок
        await msg.answer(f"❌ Ошибка обработки: {str(e)}")
        log_action(f"Error processing photo: {msg.from_user.id}, error: {str(e)}")

# ---------- ОБРАТНАЯ СВЯЗЬ ----------
@dp.message_handler(lambda m: m.text == "✏️ Отправить отзыв")
async def start_feedback(msg: types.Message):
    """
    Начало написания отзыва
    Убирает клавиатуру, чтобы пользователь не нажимал кнопки во время ввода
    """
    if user_mode.get(msg.from_user.id) != "feedback_menu":
        return

    user_mode[msg.from_user.id] = "feedback_typing"
    await msg.answer(
        "✍️ Напиши отзыв ОДНИМ сообщением.\n"
        "Кнопки сейчас использовать нельзя.",
        reply_markup=types.ReplyKeyboardRemove()  # Убираем клавиатуру
    )

@dp.message_handler(lambda m: m.text == "📄 Посмотреть мои отзывы")
async def view_my_feedback(msg: types.Message):
    """Просмотр собственных отзывов пользователя"""
    if not os.path.exists(FEEDBACK_FILE):
        await msg.answer("У вас пока нет отзывов.")
        return
    
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        all_feedback = f.readlines()
    
    # Фильтрация по user_id
    my_feedback = []
    for fb in all_feedback:
        parts = fb.strip().split("||")
        if len(parts) >= 3:
            if parts[1] == str(msg.from_user.id):  # Сравниваем user_id
                my_feedback.append(fb)
    
    if not my_feedback:
        await msg.answer("У вас пока нет отзывов.")
    else:
        # Берем последние 10 отзывов
        last_feedback = my_feedback[-10:]
        feedback_text = "📝 Ваши отзывы:\n\n"
        
        for fb in last_feedback:
            parts = fb.strip().split("||")
            if len(parts) >= 3:
                timestamp = parts[0]
                text = parts[2]
                # Форматируем дату
                date_part = timestamp.split(" ")[0] if " " in timestamp else timestamp
                feedback_text += f"📅 {date_part}: {text}\n"
        
        # Обрезаем если слишком длинный
        if len(feedback_text) > 4000:
            feedback_text = feedback_text[:4000] + "\n\n... (отзывы обрезаны)"
        await msg.answer(feedback_text)

@dp.message_handler(lambda m: user_mode.get(m.from_user.id) == "feedback_typing")
async def save_user_feedback(msg: types.Message):
    if not msg.text or len(msg.text.strip()) < 5:
        await msg.answer("❌ Отзыв слишком короткий. Минимум 5 символов.")
        return
    
    feedback_text = msg.text.strip()
    user_id = msg.from_user.id
    
    # 1. АНАЛИЗИРУЕМ отзыв
    analysis = feedback_analyzer.analyze_feedback(feedback_text, user_id)
    category = analysis["main_category"]  # Получаем категорию
    
    # 2. ОТПРАВЛЯЕМ УВЕДОМЛЕНИЕ (НЕ проверяем is_critical_feedback!)
    # Используем НОВЫЙ метод с category
    if feedback_analyzer.should_send_notification(category, user_id, feedback_text):
        await send_admin_notification(bot, user_id, feedback_text, analysis)
    
    # 3. СОХРАНЯЕМ отзыв
    save_feedback(user_id, feedback_text)
    
    # 4. УЧИМ модель на этом отзыве
    feedback_analyzer.learn_from_feedback(feedback_text, user_id, category)
    
    # 5. ВОЗВРАЩАЕМ в меню обратной связи
    user_mode[user_id] = "feedback_menu"
    
    # 6. ПОКАЗЫВАЕМ результат пользователю
    category_names = {
        "bug": "сообщение об ошибке",
        "praise": "благодарность", 
        "complaint": "жалоба",
        "suggestion": "предложение",
        "question": "вопрос",
        "unknown": "отзыв"
    }
    
    response = f"{analysis['emoji']} *Отзыв сохранён!*\n\n"
    response += f"Определено как: **{category_names.get(category, 'отзыв')}**\n"
    
    if analysis['confidence'] > 0:
        response += f"Уверенность анализа: {analysis['confidence']*100:.0f}%\n\n"
    
    # Разные ответы для разных категорий
    if category == "bug":
        response += "🐛 Спасибо за сообщение об ошибке! Мы уже получили уведомление и скоро всё починим."
    elif category == "complaint":
        response += "❌ Сожалеем о проблеме! Мы уже изучаем вашу жалобу."
    elif category == "praise":
        response += "✅ Спасибо за добрые слова! Рады, что вам нравится!"
    elif category == "suggestion":
        response += "💡 Отличное предложение! Мы его рассмотрим."
    elif category == "question":
        response += "❓ Спасибо за вопрос! Мы постараемся ответить."
    else:
        response += "📝 Спасибо за обратную связь!"
    
    # Всегда показываем приоритет (так как is_critical_feedback теперь всегда True)
    response += "\n\n⚡ Ваш отзыв был отмечен как приоритетный!"
    
    await msg.answer(response, parse_mode="Markdown", reply_markup=feedback_menu())


@dp.message_handler(commands=["notify_settings"])
async def notification_settings(msg: types.Message):
    """Настройки уведомлений (только для админа)"""
    if msg.from_user.id != ADMIN_ID:
        await msg.answer("⛔ Доступно только администратору")
        return
    
    settings = feedback_analyzer.notification_settings
    
    settings_text = f"""
🔔 *Настройки уведомлений*

*Общий статус:* {'✅ ВКЛЮЧЕНЫ' if settings['enabled'] else '🔕 ВЫКЛЮЧЕНЫ'}

*Категории:*
"""
    
    for category, enabled in settings["categories"].items():
        status = "✅" if enabled else "❌"
        category_names = {
            "bug": "🐛 Баги",
            "complaint": "❌ Жалобы", 
            "praise": "✅ Похвалы",
            "suggestion": "💡 Предложения",
            "question": "❓ Вопросы"
        }
        name = category_names.get(category, category)
        settings_text += f"{status} {name}\n"
    
    settings_text += """

*Команды управления:*
/notify_status - Текущий статус
/notify_test - Тестовое уведомление
"""
    
    await msg.answer(settings_text, parse_mode="Markdown")

@dp.message_handler(commands=["notify_status"])
async def notification_status(msg: types.Message):
    """Статус уведомлений"""
    if msg.from_user.id != ADMIN_ID:
        return
    
    settings = feedback_analyzer.notification_settings
    
    text = "🔔 *Статус уведомлений*\n\n"
    text += f"📢 Общий статус: {'✅ ВКЛЮЧЕНЫ' if settings['enabled'] else '🔕 ВЫКЛЮЧЕНЫ'}\n\n"
    text += "📋 Категории:\n"
    
    for category, enabled in settings["categories"].items():
        emoji = "✅" if enabled else "❌"
        text += f"{emoji} {category}: {'вкл' if enabled else 'выкл'}\n"
    
    await msg.answer(text, parse_mode="Markdown")
    

@dp.message_handler(commands=["notify_test"])
async def test_notification_cmd(msg: types.Message):
    if msg.from_user.id != ADMIN_ID:
        return
    await send_test_notification(msg)
    
# ---------- АДМИН: ПРОСМОТР ВСЕХ ОТЗЫВОВ ----------
@dp.message_handler(lambda m: m.text == "📄 Все отзывы" and m.from_user.id == ADMIN_ID)
async def view_all_feedback(msg: types.Message):
    """
    Просмотр всех отзывов (только для администратора)
    Разбивка на части если отзывов много
    """
    lines = read_feedback(last_n=20)
    
    if not lines:
        await msg.answer("Отзывов пока нет.")
    else:
        feedback_text = "📋 Все отзывы:\n\n" + "".join(lines)
        if len(feedback_text) > 4000:
            # Разбиваем на части по 4000 символов
            parts = [feedback_text[i:i+4000] for i in range(0, len(feedback_text), 4000)]
            for part in parts:
                await msg.answer(part)
        else:
            await msg.answer(feedback_text)

# ---------- КОМАНДА /HELP ----------
@dp.message_handler(commands=["help"])
async def help_command(msg: types.Message):
    """
    Обработчик команды /help
    Краткая справка с сохранением текущего меню
    """
    help_text = """
🤖 *Помощь по боту TimeColor:*

*Основные команды:*
/start - Главное меню
/stop - Завершить сессию (сбросить все данные)
/help - Эта справка

*Как пользоваться:*
1. Нажмите "📸 Обработать фото"
2. Выберите стиль обработки из предложенных
3. Отправьте фото
4. Получите обработанное фото!

*Обратная связь:*
- Нажмите "💬 Обратная связь" чтобы оставить отзыв

*Советы:*
- Для черно-белых фото автоматически применяется раскрашивание
- Результат зависит от исходного качества фото
"""
    
    # Сохраняем текущий режим для возврата
    current_mode = user_mode.get(msg.from_user.id)
    
    # Отправляем help
    await msg.answer(help_text, parse_mode="Markdown")
    
    # Возвращаем в соответствующее меню
    if current_mode == "photo":
        await msg.answer("↩️ Чтобы вернуться к выбору стиля, используйте меню ниже", reply_markup=styles_menu())
    elif current_mode == "feedback_menu":
        await msg.answer("↩️ Чтобы вернуться к обратной связи, используйте меню ниже", reply_markup=feedback_menu())
    elif current_mode == "admin":
        await msg.answer("↩️ Чтобы вернуться в админ-панель, используйте меню ниже", reply_markup=admin_menu())
    else:
        await msg.answer("↩️ Чтобы вернуться в главное меню, используйте меню ниже", reply_markup=main_menu(msg.from_user.id))

# ---------- FALLBACK ОБРАБОТЧИК ----------
@dp.message_handler()
async def fallback(msg: types.Message):
    """
    Обработчик всех остальных сообщений
    Предоставляет контекстно-зависимые подсказки
    """
    user_id = msg.from_user.id
    current_mode = user_mode.get(user_id)
    
    # Игнорируем команды, которые уже обработаны
    if msg.text == "❓ Помощь":
        return
    
    # Контекстные подсказки
    if current_mode == "photo":
        await msg.answer("❌ Отправь фото или выбери стиль из меню.")
    elif current_mode == "feedback_typing":
        await msg.answer("❌ Отправляй только текст отзыва. Нажми /start если передумал.")
    elif current_mode == "feedback_menu":
        await msg.answer("Используй кнопки меню 👇", reply_markup=feedback_menu())
    elif current_mode == "admin":
        await msg.answer("Используй кнопки админ-панели 👇", reply_markup=admin_menu())
    else:
        # Во всех остальных случаях показываем главное меню
        await msg.answer("Используй меню 👇", reply_markup=main_menu(user_id))

# =============== ЗАПУСК БОТА ===============
if __name__ == "__main__":
    """
    Точка входа в программу
    Инициализация файлов и запуск бота
    """
    print("=" * 50)
    print("🤖 TimeColor Bot запускается...")
    print(f"👑 Администратор: ID {ADMIN_ID}")
    print("=" * 50)
    
    # Создание необходимых файлов если их нет
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            f.write("# Файл отзывов пользователей\n")
            f.write("# Формат: timestamp||user_id||text\n")
        print(f"✅ Создан файл отзывов: {FEEDBACK_FILE}")
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# Логи работы бота\n")
            f.write("# Формат: [timestamp] сообщение\n")
        print(f"✅ Создан файл логов: {LOG_FILE}")
    
    if not os.path.exists(PHOTO_STATS_FILE):
        with open(PHOTO_STATS_FILE, "w", encoding="utf-8") as f:
            f.write("# Статистика обработки фото\n")
            f.write("# Формат: timestamp|user_id|style\n")
        print(f"✅ Создан файл статистики: {PHOTO_STATS_FILE}")
    
    print("🔄 Бот запущен и ожидает сообщений...")
    print("=" * 50)
    
    # Запуск long-polling (бесконечный цикл ожидания сообщений)
    executor.start_polling(dp, skip_updates=True)