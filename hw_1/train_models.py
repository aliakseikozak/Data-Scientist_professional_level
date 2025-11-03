import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# LightGBM и XGBoost
import lightgbm as lgb
import xgboost as xgb

def load_data(path_train='train.json', path_test='test.json'):
    """Загрузка train и test данных"""
    try:
        with open(path_train, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(path_test, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None, None, None

    train_texts = [item['ingredients'] for item in train_data]
    train_labels = [item['cuisine'] for item in train_data]
    test_texts = [item['ingredients'] for item in test_data]
    test_labels = [item['cuisine'] for item in test_data]

    print(f"Размер обучающего набора: {len(train_texts)}")
    print(f"Размер тестового набора: {len(test_texts)}")

    return train_texts, train_labels, test_texts, test_labels

def evaluate_model(model, X_test, y_test, model_name="Модель", label_encoder=None):
    """Оценка модели"""
    try:
        y_pred = model.predict(X_test)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        print(f"\n{model_name} — Результаты:")
        print(f"  Точность (Accuracy): {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        return accuracy
    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        return 0

def train_models(train_texts, train_labels, test_texts, test_labels, save_files=True):
    """Обучение моделей, подбор гиперпараметров и ансамбль"""
    try:
        print("\nСоздание TF-IDF векторизатора с n-граммами (1,3)...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1,3),
            max_features=30000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        print(f"Размер словаря: {len(vectorizer.vocabulary_)}")
        print(f"Размер матрицы обучения: {X_train.shape}")
        print(f"Размер матрицы теста: {X_test.shape}")

        if save_files:
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            print("Векторизатор сохранён в vectorizer.pkl")

        models_results = []

        # MultinomialNB
        print("\nОбучение Multinomial Naive Bayes...")
        nb_grid = GridSearchCV(MultinomialNB(), {'alpha':[0.1,0.5,1.0]}, cv=5, scoring='accuracy', n_jobs=-1)
        nb_grid.fit(X_train, train_labels)
        best_nb = nb_grid.best_estimator_
        acc = evaluate_model(best_nb, X_test, test_labels, "Multinomial Naive Bayes")
        models_results.append(('MultinomialNB', best_nb, acc, nb_grid.best_params_))

        # LogisticRegression
        print("\nОбучение Logistic Regression...")
        lr_grid = GridSearchCV(
            LogisticRegression(max_iter=3000, solver='saga', multi_class='multinomial', random_state=42),
            {'C':[0.01,0.1,1,10,50,100]},  
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        lr_grid.fit(X_train, train_labels)
        best_lr = lr_grid.best_estimator_
        acc = evaluate_model(best_lr, X_test, test_labels, "Logistic Regression")
        models_results.append(('LogisticRegression', best_lr, acc, lr_grid.best_params_))

        # RandomForest
        print("\nОбучение Random Forest...")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {'n_estimators':[100,200], 'max_depth':[None,10,20]}, cv=5, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(X_train, train_labels)
        best_rf = rf_grid.best_estimator_
        acc = evaluate_model(best_rf, X_test, test_labels, "Random Forest")
        models_results.append(('RandomForest', best_rf, acc, rf_grid.best_params_))

        # LightGBM
        print("\nОбучение LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=256,   
            max_depth=16,     
            boosting_type='gbdt',
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(X_train, train_labels)
        acc = evaluate_model(lgb_model, X_test, test_labels, "LightGBM")
        models_results.append(('LightGBM', lgb_model, acc, None))     

        # Ансамбль: Logistic + LightGBM
        print("\nОбучение ансамбля (VotingClassifier)...")
        ensemble = VotingClassifier(estimators=[
            ('lr', best_lr),
            ('lgb', lgb_model)
        ], voting='soft', n_jobs=-1)
        ensemble.fit(X_train, train_labels)
        acc = evaluate_model(ensemble, X_test, test_labels, "Ensemble")
        models_results.append(('Ensemble', ensemble, acc, None))

        # Лучшая модель
        best_model = max(models_results, key=lambda x:x[2])[1]
        best_model_name = max(models_results, key=lambda x:x[2])[0]
        print(f"\nЛучшая модель: {best_model_name}")

        if save_files:
            with open('model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            print("Модель сохранена в model.pkl")

        return best_model, vectorizer

    except Exception as e:
        print(f"Ошибка при обучении моделей: {e}")
        return None, None
