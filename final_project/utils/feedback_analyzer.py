# feedback_analyzer.py
import re
import json
import os
from collections import defaultdict, Counter

class SimpleFeedbackAnalyzer:
    """Простой анализатор отзывов на основе ключевых слов"""
    
    def __init__(self):
        self.stats_file = "feedback_stats.json"
        self.settings_file = "notification_settings.json"
        
        # Ключевые слова
        self.keyword_rules = {
            "bug": [
                "ошибка", "баг", "глюк", "глючит", "глюкает",
                "не работает", "сломал", "падает", "вылетает", 
                "тормозит", "зависает", "завис", "вылет", "краш",
                "не грузит", "не загружает", "не открывает"
            ],
            "praise": [
                "спасибо", "отличный", "супер", "класс", "круто",
                "нравится", "хорошо", "удобный", "понравилось", 
                "люблю", "отлично", "прекрасно", "замечательно",
                "восхитительно", "великолепно", "шикарно", "здорово"
            ],
            "complaint": [
                "плохо", "ужасно", "кошмар", "разочарован",
                "неудобно", "сложно", "непонятно", "долго",
                "медленно", "тупит", "тормоз", "лагает",
                "неудобный", "сложный", "непонятный", "медленный"
            ],
            "suggestion": [
                "добавьте", "хочу", "можно", "было бы",
                "сделайте", "реализуйте", "хотелось бы",
                "предлагаю", "рекомендую", "советую",
                "не хватает", "мало", "больше", "хотя бы"
            ],
            "question": [
                "как", "почему", "что", "где",
                "можно ли", "возможно ли", "какой",
                "какая", "какое", "какие", "сколько",
                "зачем", "откуда", "куда", "когда"  
            ]
        }
        
        # Эмодзи
        self.category_emojis = {
            "bug": "🐛",
            "praise": "✅", 
            "complaint": "❌",
            "suggestion": "💡",
            "question": "❓",
            "unknown": "📝"
        }
        
        # Настройки уведомлений
        self.notification_settings = {
            "categories": {
                "bug": True,
                "complaint": True,
                "praise": False,
                "suggestion": False,
                "question": False
            }
        }

        
        # Инициализируем статистику
        self.stats = self._create_empty_stats()
        self.load_stats()
        
        self.load_settings()  # Загружаем настройки
    
    def _create_empty_stats(self):
        """Создает пустую статистику"""
        return {
            "total_feedbacks": 0,
            "category_counts": defaultdict(int),
            "word_frequencies": defaultdict(lambda: defaultdict(int)),
            "user_feedback_counts": defaultdict(int)
        }
        
    def load_settings(self):
        """Загружает настройки уведомлений"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # Обновляем настройки
                self.notification_settings.update(loaded_settings)
                print(f"[NOTIFICATIONS] Настройки загружены: enabled={self.notification_settings['enabled']}")
            except Exception as e:
                print(f"[NOTIFICATIONS ERROR] Ошибка загрузки настроек: {e}")
    
    def load_stats(self):
        """Загружает статистику из файла"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Сбрасываем статистику
                self.stats = self._create_empty_stats()
                
                # Загружаем данные
                self.stats["total_feedbacks"] = data.get("total_feedbacks", 0)
                
                # category_counts
                if "category_counts" in data:
                    for category, count in data["category_counts"].items():
                        self.stats["category_counts"][category] = count
                
                # word_frequencies
                if "word_frequencies" in data:
                    for word, categories in data["word_frequencies"].items():
                        for category, count in categories.items():
                            self.stats["word_frequencies"][word][category] = count
                
                # user_feedback_counts
                if "user_feedback_counts" in data:
                    for user_id, count in data["user_feedback_counts"].items():
                        self.stats["user_feedback_counts"][user_id] = count
                        
                print(f"[FEEDBACK] Загружено {self.stats['total_feedbacks']} отзывов")
                
            except Exception as e:
                print(f"[FEEDBACK ERROR] Ошибка загрузки статистики: {e}")
                self.stats = self._create_empty_stats()
        else:
            print(f"[FEEDBACK] Файл статистики не найден, создаем новый")
            self.save_stats()
    
    def save_stats(self):
        """Сохраняет статистику в файл"""
        try:
            # Конвертируем defaultdict в обычные dict
            stats_to_save = {
                "total_feedbacks": self.stats["total_feedbacks"],
                "category_counts": dict(self.stats["category_counts"]),
                "word_frequencies": {},
                "user_feedback_counts": dict(self.stats["user_feedback_counts"])
            }
            
            # Конвертируем word_frequencies
            for word, categories in self.stats["word_frequencies"].items():
                if categories:  # Если есть данные
                    stats_to_save["word_frequencies"][word] = dict(categories)
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
                
            #print(f"[FEEDBACK] Статистика сохранена ({self.stats['total_feedbacks']} отзывов)")
            
        except Exception as e:
            print(f"[FEEDBACK ERROR] Ошибка сохранения статистики: {e}")
    
    def clean_text(self, text):
        """Очищает текст"""
        text = text.lower()
        
        # Заменяем знаки пунктуации на пробелы
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        
        # Расширенный список стоп-слов
        stop_words = {
            "это", "очень", "просто", "так", "то",
            "что", "как", "вот", "ну", "же", "ли",
            "бы", "ведь", "вроде", "даже", "еще",
            "уже", "только", "просто", "сам", "сама"
        }
        
        # Оставляем слова длиной от 2 символов
        words = [w for w in words if len(w) >= 2 and w not in stop_words]
        
        return words
    
    def analyze_feedback(self, text, user_id=None):
        """Анализирует отзыв"""
        words = self.clean_text(text)
        original_text = text.lower()
        
        category_scores = defaultdict(int)
        
        # Сначала проверяем фразы (сочетания слов)
        phrase_matches = {
            "bug": ["не работает", "не грузит", "не открывает", "вылетает при"],
            "complaint": ["слишком долго", "очень долго", "очень медленно"],
            "suggestion": ["было бы", "хотелось бы", "не хватает"],
            "question": ["можно ли", "возможно ли", "как сделать", "как сохранить"]
        }
        
        for category, phrases in phrase_matches.items():
            for phrase in phrases:
                if phrase in original_text:
                    category_scores[category] += 2  # Фразы важнее отдельных слов
        
        # Затем проверяем отдельные слова
        for word in words:
            for category, keywords in self.keyword_rules.items():
                # Проверяем точное совпадение или вхождение
                if any(keyword == word or word.startswith(keyword[:3]) for keyword in keywords):
                    category_scores[category] += 1
        
        # Определяем категорию
        if category_scores:
            main_category = max(category_scores.items(), key=lambda x: x[1])[0]
            confidence = category_scores[main_category] / (len(words) + 2)  # +2 для нормализации
        else:
            main_category = "unknown"
            confidence = 0
        
        # Повышаем уверенность для явных случаев
        if main_category == "unknown" and "?" in text:
            main_category = "question"
            confidence = 0.7
        elif main_category == "unknown" and any(word in ["пожалуйста", "спасибо"] for word in words):
            main_category = "praise"
            confidence = 0.6
        
        return {
            "main_category": main_category,
            "confidence": min(1.0, confidence),  # Ограничиваем 100%
            "word_count": len(words),
            "emoji": self.category_emojis.get(main_category, "📝")
        }
    
    def learn_from_feedback(self, text, user_id=None, confirmed_category=None):
        """Учимся на отзыве"""
        # Анализируем
        analysis = self.analyze_feedback(text, user_id)
        
        # Определяем категорию
        category = confirmed_category if confirmed_category else analysis["main_category"]
        
        #print(f"[FEEDBACK] Обучение на категории: {category}")
        
        # Обновляем статистику (defaultdict сам создаст ключи)
        self.stats["total_feedbacks"] += 1
        self.stats["category_counts"][category] += 1
        
        if user_id:
            self.stats["user_feedback_counts"][str(user_id)] += 1
        
        # Сохраняем слова
        words = self.clean_text(text)
        for word in words:
            self.stats["word_frequencies"][word][category] += 1
        
        # Сохраняем
        self.save_stats()
        
        return analysis
    
    def is_critical_feedback(self, category):
        """Проверяет, критичный ли отзыв"""
        #return category in ["bug", "complaint"]
        return True

    def save_settings(self):
        """Сохраняет настройки уведомлений"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.notification_settings, f, ensure_ascii=False, indent=2)
            print(f"[NOTIFICATIONS] Настройки сохранены")
        except Exception as e:
            print(f"[NOTIFICATIONS ERROR] Ошибка сохранения: {e}")
    
    def is_category_enabled(self, category):
        """Проверяет, включена ли категория"""
        # Просто проверяем настройку категории (общего статуса больше нет)
        return self.notification_settings["categories"].get(category, False)

    def are_all_categories_enabled(self):
        """Проверяет, включены ли ВСЕ категории"""
        categories = self.notification_settings.get("categories", {})
        return all(categories.values()) if categories else False

    def are_all_categories_disabled(self):
        """Проверяет, выключены ли ВСЕ категории"""
        categories = self.notification_settings.get("categories", {})
        return not any(categories.values()) if categories else True
        
    def should_send_notification(self, category, user_id=None, feedback_text=None):
        """Проверяет, нужно ли отправлять уведомление для данной категории"""
        # Просто проверяем, включена ли категория
        return self.is_category_enabled(category)
    
    def format_notification_message(self, user_id, feedback_text, analysis):
        """Форматирует уведомление"""
        from datetime import datetime
        
        current_time = datetime.now().strftime("%H:%M:%S")
        short_text = feedback_text[:200] + "..." if len(feedback_text) > 200 else feedback_text
        
        return f"""
                    {analysis['emoji']} *УВЕДОМЛЕНИЕ*

                    👤 Пользователь: `{user_id}`
                    📊 Категория: {analysis['main_category'].upper()}
                    🕒 Время: {current_time}
                    📈 Уверенность: {analysis['confidence']*100:.0f}%

                    💬 Текст:{short_text}
                    ⚡ Требует внимания!
                    """
    def get_stats_summary(self):
        """Возвращает статистику"""
        total = self.stats["total_feedbacks"]
        
        if total == 0:
            return "Нет отзывов"
        
        result = f"📊 Всего отзывов: {total}\n\n"
        result += "📈 Распределение:\n"
        
        for category, count in sorted(
            self.stats["category_counts"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percent = (count / total) * 100
            emoji = self.category_emojis.get(category, "📝")
            result += f"{emoji} {category}: {count} ({percent:.1f}%)\n"
        
        return result
        
    def get_insights(self):
        """Возвращает аналитические инсайты"""
        total = self.stats["total_feedbacks"]
        
        if total < 10:
            return "Нужно больше отзывов для анализа (минимум 10)"
        
        insights = []
        
        # Анализ преобладающих категорий
        praise_count = self.stats["category_counts"].get("praise", 0)
        complaint_count = self.stats["category_counts"].get("complaint", 0)
        bug_count = self.stats["category_counts"].get("bug", 0)
        
        if praise_count > complaint_count * 2:
            insights.append("✅ Пользователи довольны! Позитивных отзывов в 2+ раза больше")
        elif complaint_count > praise_count:
            insights.append("⚠️ Много жалоб. Стоит обратить внимание на улучшение сервиса")
        
        if bug_count > total * 0.1:  # Если более 10% отзывов - об ошибках
            insights.append(f"🐛 {bug_count} сообщений об ошибках ({bug_count/total*100:.1f}%) - требуется проверка")
        
        # Анализ активности
        unique_users = len(self.stats["user_feedback_counts"])
        if unique_users > 0:
            avg_feedbacks = total / unique_users
            insights.append(f"👥 {unique_users} уникальных пользователей, в среднем по {avg_feedbacks:.1f} отзыва")
        
        # Популярные запросы
        suggestion_count = self.stats["category_counts"].get("suggestion", 0)
        if suggestion_count > 0:
            insights.append(f"💡 {suggestion_count} предложений по улучшению")
        
        return "\n".join(insights) if insights else "Пока нет значимых инсайтов"

# Глобальный объект
feedback_analyzer = SimpleFeedbackAnalyzer()