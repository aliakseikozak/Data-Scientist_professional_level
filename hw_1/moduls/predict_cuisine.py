import sqlite3
import pickle
import re
import numpy as np

DB_PATH = "db/recipes.db"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def clean_text(text):
    """Очистка текста от мусора, чисел и символов"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model_and_vectorizer():
    """Загрузка модели и TF-IDF векторизатора"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Модель и векторизатор загружены.")
    return model, vectorizer


def save_prediction_to_db(ingredients, predicted_cuisine, probabilities):
    """Сохранение предсказания в базу данных"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredients TEXT,
            predicted_cuisine TEXT,
            probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO predictions (input_text, predicted_cuisine, probability)
        VALUES (?, ?, ?)
    """, (ingredients, predicted_cuisine, float(max(probabilities))))

    conn.commit()
    conn.close()
    print("💾 Предсказание сохранено в БД.")


def show_recent_predictions(limit=5):
    """Показ последних предсказаний"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT input_text, predicted_cuisine, probability, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("📭 История пуста.")
        return

    print("\n🕘 Последние предсказания:")
    for row in rows:
        print(f"🧾 {row[3]} | {row[1]} ({row[2]:.4f}) ← {row[0]}")


def predict_cuisine(input_text):
    """Основная функция предсказания"""
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(input_text)
    X = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(X)[0]
    cuisines = model.classes_
    top_idx = np.argmax(probabilities)
    predicted_cuisine = cuisines[top_idx]

    # Вывод
    print(f"\n🥘 Предсказанная кухня: {predicted_cuisine}")
    print("🔢 Топ-5 вероятностей:")
    top_indices = np.argsort(probabilities)[::-1][:5]
    for idx in top_indices:
        print(f"  {cuisines[idx]}: {probabilities[idx]:.4f}")

    save_prediction_to_db(input_text, predicted_cuisine, probabilities.tolist())
    show_recent_predictions()