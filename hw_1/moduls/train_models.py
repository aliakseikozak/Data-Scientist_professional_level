import sqlite3
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import lightgbm as lgb

DB_PATH = "db/recipes.db"


# ===============================
# 🧹 Функция очистки текста
# ===============================
def clean_text(text):
    """Очистка текста от лишних символов"""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ===============================
# 📥 Загрузка данных из SQLite
# ===============================
def load_data_from_db():
    """Загружает обучающие и тестовые данные из БД"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ingredients, cuisine FROM recipes")
    data = cursor.fetchall()
    conn.close()

    if not data:
        raise ValueError("❌ В таблице recipes нет данных!")

    texts = [clean_text(row[0]) for row in data if row[0]]
    labels = [row[1] for row in data if row[1]]

    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]

    print(f"📊 Загружено из БД: {len(texts)} записей")
    print(f"  → Обучение: {len(train_texts)}")
    print(f"  → Тест: {len(test_texts)}")
    print(f"✅ Пример текста:\n{train_texts[0][:200]}")

    return train_texts, train_labels, test_texts, test_labels


# ===============================
# 📈 Оценка модели
# ===============================
def evaluate_model(model, X_test, y_test, model_name="Модель"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"\n📈 {model_name} — результаты:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    return accuracy


# ===============================
# 🧠 Обучение моделей
# ===============================
def train_models(save_files=True):
    """Обучение моделей, подбор параметров и выбор лучшей"""
    train_texts, train_labels, test_texts, test_labels = load_data_from_db()

    print("\nСоздание TF-IDF векторизатора (1–3-граммы)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=30000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
    print(f"Матрица обучения: {X_train.shape}")
    print(f"Матрица теста: {X_test.shape}")

    if save_files:
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        print("✅ Векторизатор сохранён: models/vectorizer.pkl")

    models_results = []

    # Naive Bayes
    print("\n🧠 Обучение Multinomial Naive Bayes...")
    nb_grid = GridSearchCV(MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}, cv=5, scoring='accuracy', n_jobs=-1)
    nb_grid.fit(X_train, train_labels)
    best_nb = nb_grid.best_estimator_
    acc = evaluate_model(best_nb, X_test, test_labels, "Multinomial Naive Bayes")
    models_results.append(('MultinomialNB', best_nb, acc, nb_grid.best_params_))

    # Logistic Regression
    print("\n⚙️ Обучение Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=3000, solver='saga', multi_class='multinomial', random_state=42),
        {'C': [0.01, 0.1, 1, 10, 50, 100]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train, train_labels)
    best_lr = lr_grid.best_estimator_
    acc = evaluate_model(best_lr, X_test, test_labels, "Logistic Regression")
    models_results.append(('LogisticRegression', best_lr, acc, lr_grid.best_params_))

    # Random Forest
    print("\n🌲 Обучение Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X_train, train_labels)
    best_rf = rf_grid.best_estimator_
    acc = evaluate_model(best_rf, X_test, test_labels, "Random Forest")
    models_results.append(('RandomForest', best_rf, acc, rf_grid.best_params_))

    # LightGBM
    print("\n💡 Обучение LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.1, num_leaves=256,
        max_depth=16, boosting_type='gbdt', random_state=42, n_jobs=-1
    )
    lgb_model.fit(X_train, train_labels)
    acc = evaluate_model(lgb_model, X_test, test_labels, "LightGBM")
    models_results.append(('LightGBM', lgb_model, acc, None))

    # Ensemble
    print("\n🤝 Обучение ансамбля (VotingClassifier)...")
    ensemble = VotingClassifier(
        estimators=[('lr', best_lr), ('lgb', lgb_model)], voting='soft', n_jobs=-1
    )
    ensemble.fit(X_train, train_labels)
    acc = evaluate_model(ensemble, X_test, test_labels, "Ensemble")
    models_results.append(('Ensemble', ensemble, acc, None))

    best_model = max(models_results, key=lambda x: x[2])[1]
    best_name = max(models_results, key=lambda x: x[2])[0]
    print(f"\n🏆 Лучшая модель: {best_name}")

    if save_files:
        with open("models/model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("✅ Модель сохранена: models/model.pkl")

    return best_model, vectorizer
