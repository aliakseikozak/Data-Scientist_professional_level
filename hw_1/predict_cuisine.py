import pickle
import re
import matplotlib.pyplot as plt

def clean_ingredients(ingredients):
    """Очистка ингредиентов: приведение к нижнему регистру, удаление чисел, дробей, скобок, спецсимволов и лишних пробелов"""
    cleaned = []
    for ing in ingredients:
        try:
            ing = ing.lower()
            ing = re.sub(r'[\(\[].*?[\)\]]', '', ing)
            ing = re.sub(r'[^a-zA-Z\s]', '', ing)
            ing = re.sub(r'\s+', ' ', ing).strip()
            if ing:
                cleaned.append(ing)
        except Exception as e:
            print(f"Ошибка при очистке ингредиента '{ing}': {e}")
    return cleaned

def predict_cuisine(ingredients_list, show_plot=True):
    """Предсказание кухни и визуализация топ-3"""
    try:
        # Загрузка модели и векторизатора
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке модели или векторизатора: {e}")
        return None, None

    cleaned_ingredients = clean_ingredients(ingredients_list)
    ingredients_str = ' '.join(cleaned_ingredients)
    X = vectorizer.transform([ingredients_str])

    try:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
        prob_sorted = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None, None

    # Вывод топ-3
    top3 = prob_sorted[:3]
    print("\nИсходные ингредиенты:", ingredients_list)
    print("Очистка ингредиентов:", cleaned_ingredients)
    print("\nТоп-3 предсказанных кухни:")
    for cuisine, prob in top3:
        print(f"{cuisine:15s}: {prob*100:.2f}%")

    # Графическая визуализация
    if show_plot:
        cuisines, probs = zip(*top3)
        plt.figure(figsize=(6, 3))
        plt.barh(cuisines[::-1], [p*100 for p in probs[::-1]], color='orange')
        plt.xlabel('Вероятность, %')
        plt.title('Топ-3 предсказанные кухни')
        plt.xlim(0, 100)
        for i, v in enumerate(probs[::-1]):
            plt.text(v*100 + 1, i, f"{v*100:.1f}%", va='center')
        plt.show()

    return prediction, prob_sorted
