import streamlit as st
import sqlite3
import pickle
import re
import numpy as np
import pandas as pd
from datetime import datetime

DB_PATH = "db/recipes.db"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


# --- 🔹 Вспомогательные функции ---
@st.cache_resource
def load_model_and_vectorizer():
    """Загрузка модели и TF-IDF векторизатора"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def clean_text(text):
    """Очистка текста"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_prediction_to_db(ingredients, predicted_cuisine, probability):
    """Сохранение результата в SQLite"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Таблица создаётся только если не существует
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            predicted_cuisine TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        INSERT INTO predictions (input_text, predicted_cuisine, probability)
        VALUES (?, ?, ?)
    """, (ingredients, predicted_cuisine, float(probability)))
    conn.commit()
    conn.close()


def get_recent_predictions(limit=10):
    """Получение последних N записей"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT input_text, predicted_cuisine, probability, timestamp FROM predictions ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


# --- ⚙️ Интерфейс Streamlit ---
st.set_page_config(page_title="Cuisine Classifier", page_icon="🍽️", layout="centered")

st.title("🍳 Классификация кухни по ингредиентам")
st.write("Введите ингредиенты блюда, и модель определит, к какой кухне оно относится.")

model, vectorizer = load_model_and_vectorizer()

# --- Ввод ингредиентов ---
user_input = st.text_area(
    "🧂 Введите ингредиенты через запятую:",
    height=100,
    placeholder="chicken, garlic, soy sauce, ginger"
)

if st.button("Предсказать кухню"):
    if not user_input.strip():
        st.warning("Введите хотя бы один ингредиент!")
    else:
        cleaned = clean_text(user_input)
        X = vectorizer.transform([cleaned])
        probabilities = model.predict_proba(X)[0]
        cuisines = model.classes_
        top_idx = np.argmax(probabilities)
        predicted_cuisine = cuisines[top_idx]
        top_probability = float(probabilities[top_idx])  # <-- сохраняем только максимум

        # 🔹 Сохранение в БД
        save_prediction_to_db(user_input, predicted_cuisine, top_probability)

        # 🔹 Вывод результата
        st.success(f"🥘 **Предсказанная кухня:** {predicted_cuisine}")

        st.subheader("🔢 Топ-5 вероятностей:")
        top_indices = np.argsort(probabilities)[::-1][:5]
        prob_data = {
            "Кухня": [cuisines[i] for i in top_indices],
            "Вероятность": [float(probabilities[i]) for i in top_indices]
        }
        st.bar_chart(pd.DataFrame(prob_data).set_index("Кухня"))

# --- История предсказаний ---
st.markdown("---")
st.subheader("🕘 История последних предсказаний")
history = get_recent_predictions()
if not history.empty:
    st.dataframe(history)
else:
    st.info("История пока пуста — сделайте первое предсказание! 😋")
