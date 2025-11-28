import logging
import json
import random
import torch
import re
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
from config import Config
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import datetime
from natasha import Doc, Segmenter
from pymorphy2 import MorphAnalyzer
from dotenv import load_dotenv
import os

# ---------- Логирование ----------
now = datetime.datetime.now().strftime("%Y-%m-%d")
logging.basicConfig(
    filename=f'telegram_chat_{now}.log',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# ---------- Загрузка интентов ----------
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Убираем все "?" из паттернов
for intent in intents["intents"]:
    intent["patterns"] = [p.replace("?", "").strip() for p in intent.get("patterns", [])]

clarification_intent = next((i for i in intents['intents'] if i['tag'] == 'clarification'), None)
unknown_intent = next((i for i in intents['intents'] if i['tag'] == 'unknown'), None)

# ---------- Загрузка модели ----------
device = Config.DEVICE
data = torch.load("data.pth", map_location=device)
model = NeuralNet(data["input_size"], data["hidden_sizes"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

all_words = data['all_words']
tags = data['tags']

# ---------- Контекст ----------
user_context = {}  # {user_id: {"history": [...], "awaiting_city": intent_data}}

# ---------- Города ----------
with open("cities.json", "r", encoding="utf-8") as f:
    cities = json.load(f)["cities"]

# ---------- Map город -> страна ----------
country_map = {
    "бали": "Индонезия",
    "убуд": "Индонезия",
    "сингапур": "Сингапур",
    "хиккадува": "Шри-Ланка",
    "таиланд": "Таиланд",
    "бангкок": "Таиланд",
    "пхукет": "Таиланд",
    "сеул": "Южная Корея",
    "токио": "Япония",
    "фукуок": "Вьетнам"
}

# ---------- Пороговые настройки ----------
BASE_HIGH_CONF = 0.8
BASE_LOW_CONF = 0.5
HISTORY_LEN = 3

# ---------- Natasha + pymorphy2 для лемматизации ----------
segmenter = Segmenter()
morph = MorphAnalyzer()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # убираем знаки препинания
    text = re.sub(r'\s+', ' ', text).strip()  # убираем лишние пробелы
    doc = Doc(text)
    doc.segment(segmenter)
    lemmas = [morph.parse(token.text)[0].normal_form for token in doc.tokens]
    return " ".join(lemmas)

def find_city_or_country(sentence):
    norm_sentence = normalize_text(sentence)
    for c in cities:
        if normalize_text(c) in norm_sentence:
            return c, country_map.get(c.lower(), None)
    for country in set(country_map.values()):
        if normalize_text(country) in norm_sentence:
            return None, country
    return None, None

def match_pattern(sentence, patterns):
    """
    Сравниваем запрос с паттернами через множества слов (порядок слов не важен)
    """
    sent_tokens = set(normalize_text(sentence).split())
    for p in patterns:
        pattern_tokens = set(normalize_text(p).split())
        if pattern_tokens <= sent_tokens:
            return True
    return False

# ---------- Постфильтр ответа ----------
def normalize_text_keep_hyphen(text):
    """
    Лемматизация и приведение к нижнему регистру,
    но дефис сохраняется.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)  # сохраняем дефис, убираем остальные знаки
    text = re.sub(r"\s+", " ", text).strip()
    doc = Doc(text)
    doc.segment(segmenter)
    lemmas = [morph.parse(token.text)[0].normal_form for token in doc.tokens]
    return " ".join(lemmas)

def filter_response(intent_data, city=None, country=None, sentence=None):
    responses = intent_data.get("responses", ["Извините, не могу подсказать."])

    # ---------- Логика для visa_info ----------
    if intent_data.get("tag") == "visa_info" and sentence:
        norm_sentence = normalize_text_keep_hyphen(sentence)
        for keyword, country_name in visa_map.items():
            if keyword in norm_sentence:
                for r in responses:
                    r_norm = normalize_text_keep_hyphen(r)
                    if keyword in r_norm or normalize_text_keep_hyphen(country_name) in r_norm:
                        return r
        return intent_data.get("default_response", random.choice(responses))

    # ---------- Логика для интентов без городов/стран ----------
    if intent_data.get("tag") not in ["visa_info", "excursions_tours", "restaurants", 
                                      "hotel_info", "attractions", "transport_info", 
                                      "activities", "shopping", "weather_info",
                                      "nightlife", "currency_exchange", "taxi_service"]:
        return random.choice(responses)

    # ---------- Общая логика для остальных интентов ----------
    norm_city = normalize_text(city) if city else None
    norm_country = normalize_text(country) if country else None

    for r in responses:
        r_norm = normalize_text(r)
        if norm_city:
            city_tokens = set(norm_city.split())
            if city_tokens & set(r_norm.split()):
                return r
        if norm_country:
            country_tokens = set(norm_country.split())
            if country_tokens & set(r_norm.split()) or list(country_tokens)[0] in r_norm:
                return r

    # ---------- fallback для всех остальных интентов ----------
    return intent_data.get("default_response", random.choice(responses))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username or str(user_id)
    sentence = update.message.text.strip()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if user_id not in user_context:
        user_context[user_id] = {"history": [], "awaiting_city": None}

    user_context[user_id]["history"].append(sentence)
    if len(user_context[user_id]["history"]) > HISTORY_LEN:
        user_context[user_id]["history"].pop(0)

    city_found, country_found = find_city_or_country(sentence)

    # ---------- Если пользователь прислал уточнение города ----------
    if user_context[user_id]["awaiting_city"]:
        intent_data = user_context[user_id]["awaiting_city"]
        if city_found or country_found:
            response = filter_response(intent_data, city_found, country_found)
        else:
            response = "Пожалуйста, уточните город или страну из списка."
        user_context[user_id]["awaiting_city"] = None
        with open("chat_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} | {username} | Q: {sentence} | A: {response} | Tag: {intent_data['tag']} | Prob: 1.00 | Type: city_response\n")
        await update.message.reply_text(response)
        return

    # ---------- Модельная логика ----------
    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()
    intent_data = next((i for i in intents["intents"] if i["tag"] == tag), None)

    needs_city = tag in ["restaurants", "hotel_info", "attractions", "transport_info",
                         "activities", "shopping", "weather_info", "visa_info"]

    # ---------- Уточнение города ----------
    if needs_city and not (city_found or country_found) and prob >= BASE_HIGH_CONF:
        user_context[user_id]["awaiting_city"] = intent_data
        response = "Уточните, в каком городе или стране вас интересует это?"
        intent_type = "clarification"
        with open("chat_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} | {username} | Q: {sentence} | A: {response} | Tag: {tag} | Prob: {prob:.2f} | Type: clarification\n")
        await update.message.reply_text(response)
        return

    # ---------- Попытка совпадения с паттернами ----------
    matched_pattern = False
    for intent in intents["intents"]:
        if match_pattern(sentence, intent.get("patterns", [])):
            intent_data = intent
            tag = intent["tag"]
            matched_pattern = True
            prob = 1.0
            break

    # ---------- Постфильтр ответа ----------
    if prob < BASE_LOW_CONF and not matched_pattern:
        if tag == "unknown" or not needs_city:
            response = random.choice(unknown_intent["responses"])
            intent_type = "unknown"
        else:
            response = random.choice(clarification_intent["responses"])
            intent_type = "clarification"
    else:
        response = filter_response(intent_data, city_found, country_found)
        intent_type = "high_conf" if prob >= BASE_HIGH_CONF else "clarification"

    # ---------- Сохраняем unknown ----------
    if intent_type == "unknown":
        with open("intents.json", 'r+', encoding='utf-8') as f_json:
            data_json = json.load(f_json)
            unknown = next((i for i in data_json["intents"] if i["tag"] == "unknown"), None)
            if sentence not in unknown["patterns"]:
                unknown["patterns"].append(sentence)
                f_json.seek(0)
                json.dump(data_json, f_json, ensure_ascii=False, indent=4)
                f_json.truncate()
        with open("unknown_questions.txt", "a", encoding="utf-8") as f_txt:
            f_txt.write(f"{timestamp} | {username} | {sentence}\n")

    # ---------- Логирование ----------
    with open("chat_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} | {username} | Q: {sentence} | A: {response} | Tag: {tag} | Prob: {prob:.2f} | Type: {intent_type}\n")

    await update.message.reply_text(response)


# ---------- Команда /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я твоя туристическая Наташа. Напиши мне что-нибудь!")

# ---------- Запуск бота ----------
if __name__ == "__main__":
    load_dotenv(dotenv_path="token_bot.env")
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram бот Наташа запущен...")
    app.run_polling()
