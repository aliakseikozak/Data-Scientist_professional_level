import sqlite3
import json
import os

DB_PATH = "recipes.db"
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"


def create_tables(conn):
    cursor = conn.cursor()

    # Таблица с данными (ингредиенты и метка кухни)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cuisine TEXT,
            ingredients TEXT
        );
    """)

    # Таблица для логирования предсказаний
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            predicted_cuisine TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()


def load_json_to_db(conn, json_path, table="recipes"):
    if not os.path.exists(json_path):
        print(f"⚠️ Файл {json_path} не найден — пропуск.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cursor = conn.cursor()
    added = 0

    for item in data:
        cuisine = item.get("cuisine", "").strip()
        ingredients = item.get("ingredients", "")

        # очистим ингредиенты от двойных пробелов, переводов строк и т.д.
        ingredients = " ".join(ingredients.split()).lower().strip()

        cursor.execute(
            "INSERT INTO recipes (cuisine, ingredients) VALUES (?, ?)",
            (cuisine, ingredients)
        )
        added += 1

    conn.commit()
    print(f"✅ Загружено {added} записей из {json_path} в таблицу '{table}'.")



def init_db():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    # Проверяем, пустая ли таблица
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM recipes")
    count = cursor.fetchone()[0]

    if count == 0:
        print("📥 Таблица пуста — загружаем данные из JSON...")
        load_json_to_db(conn, TRAIN_JSON)
        load_json_to_db(conn, TEST_JSON)
    else:
        print(f"ℹ️ В таблице уже есть {count} записей — загрузка пропущена.")

    conn.close()
    print(f"✅ База данных готова: {DB_PATH}")