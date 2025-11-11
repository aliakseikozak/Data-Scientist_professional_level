# Face Verification with Siamese Network

Проект для сравнения лиц на изображениях с помощью Siamese Network на ResNet18. Используется датасет [LFW](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) и контрастная функция потерь для обучения.

## Структура проекта

```
hw_2/
├─ data/lfw_funneled/         # Папка с изображениями LFW
├─ data/photo/                # Папка с фото для тестирования на которых модель не проходила обучение
├─ data/pairsDevTrain.txt      # Файл с тренировочными парами
├─ data/pairsDevTest.txt       # Файл с тестовыми парами
├─ modules/
│  ├─ train_model.py           # Скрипт обучения модели
│  ├─ file_processing.py       # Предобработка данных для дальнейшего использования при обучение модели
│  └─ test_model.py            # Функция compare_faces для тестирования
├─ models/
│  └─ siamese_resnet_best.pth  # Сохранённые веса модели
└─ README.md
```

## Подготовка данных модели

```bash
python file_processing.py
```

## Обучение модели

```bash
python train_model.py
```

- Веса модели сохраняются в `models/siamese_resnet_best.pth`.  
- Используется early stopping по AUC на тестовом наборе.  
- Контрастная функция потерь с `margin=1.5`.  

## Тестирование / сравнение лиц

```python
from modules.test_model import compare_faces

compare_faces(
    model_path="models/siamese_resnet_best.pth",
    img_path1=r"data/lfw_funneled/David_Beckham/David_Beckham_0013.jpg",
    img_path2=r"data/lfw_funneled/David_Beckham/David_Beckham_0014.jpg",
    threshold=2.0
)
```

- Результат выводится в консоль и на графике.  
- Пути к изображениям указываются относительно корня проекта.  
- `threshold` — максимальное значение L2 distance для 100% схожести (подбирается по валидации).

## Пояснения

- **Siamese Network** — два идентичных ResNet18, выходы нормализуются L2.  
- **Контрастная потеря**:  
  \[ L = label \cdot ||x_1 - x_2||^2 + (1-label) \cdot max(0, margin - ||x_1 - x_2||)^2 \]  
- **Метрики**: Accuracy, средние расстояния положительных/отрицательных пар, AUC.

## Примечания

- Используй GPU для ускоренного обучения.  
- Размер изображений — 128×128.  
- В `modules/test_model.py` можно визуализировать любые две фотографии и увидеть результат.