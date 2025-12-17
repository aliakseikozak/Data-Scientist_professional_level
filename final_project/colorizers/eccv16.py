"""
МОДЕЛЬ ECCV16 ДЛЯ АВТОМАТИЧЕСКОГО РАСКРАШИВАНИЯ ИЗОБРАЖЕНИЙ
===========================================================
Модель из статьи "Colorful Image Colorization" (Zhang et al., ECCV 2016)
Архитектура: Полностью сверточная нейронная сеть (FCN) для предсказания
             цветности (ab-каналов) по яркости (L-каналу).

Ключевые особенности:
1. Кодировщик-декодировщик с пропускающими соединениями
2. Использование dilated convolutions для увеличения receptive field
3. Выход в пространстве вероятностей 313-мерного цветового пространства
4. Преобразование в ab-каналы через обученную матрицу перехода
"""

import torch
import torch.nn as nn
from .base_color import BaseColor  # Базовый класс для работы с LAB цветовым пространством


class ECCVGenerator(BaseColor):
    """
    ГЕНЕРАТОР ECCV16
    ================
    Наследуется от BaseColor для нормализации/денормализации LAB цветов.
    
    Архитектурные блоки:
    - model1-4: Кодировщик (downsampling 8x)
    - model5-7: Промежуточные слои с dilated convolutions
    - model8: Декодировщик (upsampling)
    - softmax + model_out: Преобразование в цветовое пространство
    """
    
    def __init__(self, norm_layer=nn.BatchNorm2d):
        """
        ИНИЦИАЛИЗАЦИЯ АРХИТЕКТУРЫ ECCV16
        
        Args:
            norm_layer: Слой нормализации (по умолчанию BatchNorm2d)
                       Можно заменить на InstanceNorm2d для style transfer
        
        Структура сети:
        Вход: 1 канал (L-канал яркости) → Выход: 2 канала (ab-каналы цветности)
        Размерность уменьшается в 8 раз в кодировщике и восстанавливается в декодировщике
        """
        super(ECCVGenerator, self).__init__()

        # ====================
        # ЭТАП 1: КОДИРОВЩИК (ENCODER)
        # Уменьшение размерности с извлечением признаков
        # ====================
        
        # Блок 1: 1 → 64 канала, downsample 2x
        # Сохраняет детали низкого уровня (границы, текстуры)
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),           # Conv: 1→64, kernel=3, stride=1, padding=1
            nn.ReLU(True),                       # Активация ReLU (inplace=True для экономии памяти)
            nn.Conv2d(64, 64, 3, 2, 1),          # Conv: 64→64, downsample 2x (stride=2)
            nn.ReLU(True), 
            norm_layer(64)                       # Batch Normalization: стабилизация обучения
        )
        # Размер: H×W → H/2 × W/2, каналы: 1 → 64

        # Блок 2: 64 → 128 каналов, downsample 2x
        # Извлекает признаки среднего уровня
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),         # Conv: 64→128
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1),        # Conv с downsample 2x
            nn.ReLU(True), 
            norm_layer(128)
        )
        # Размер: H/2 × W/2 → H/4 × W/4, каналы: 64 → 128

        # Блок 3: 128 → 256 каналов, downsample 2x
        # Извлекает признаки высокого уровня (объекты, структуры)
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),        # Conv: 128→256
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),        # Дополнительный conv для увеличения емкости
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),        # Conv с downsample 2x
            nn.ReLU(True), 
            norm_layer(256)
        )
        # Размер: H/4 × W/4 → H/8 × W/8, каналы: 128 → 256

        # Блок 4: 256 → 512 каналов (без downsampling)
        # Глубокие абстрактные признаки
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),        # Conv: 256→512
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),        # Углубление признаков
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),        # Еще один слой
            nn.ReLU(True), 
            norm_layer(512)
        )
        # Размер остается: H/8 × W/8, каналы: 256 → 512

        # ====================
        # ЭТАП 2: ПРОМЕЖУТОЧНЫЕ СЛОИ С DILATED CONVOLUTIONS
        # Увеличение receptive field без потери разрешения
        # ====================
        
        # Блок 5-6: Dilated convolutions (dilation=2)
        # Увеличивают область восприятия (receptive field) в 3 раза
        # dilation=2 означает "дырчатую" свертку с пропуском пикселей
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),  # padding=2 для сохранения размера
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.ReLU(True), 
            norm_layer(512)
        )
        
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 2, dilation=2),
            nn.ReLU(True), 
            norm_layer(512)
        )
        # Receptive field каждого пикселя охватывает большую область изображения

        # Блок 7: Обычные свертки для уточнения признаков
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),        # Возврат к обычным сверткам
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True), 
            norm_layer(512)
        )

        # ====================
        # ЭТАП 3: ДЕКОДИРОВЩИК (DECODER)
        # Восстановление размера и генерация цветности
        # ====================
        
        # Блок 8: Transposed convolution для upsampling 2x
        # 512 каналов → 256 каналов, увеличение размера в 2 раза
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Transposed conv: upscale 2x
            nn.ReLU(True),                          # kernel=4, stride=2, padding=1
            nn.Conv2d(256, 256, 3, 1, 1),           # Уточнение признаков после upsampling
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),           # Дополнительное уточнение
            nn.ReLU(True),
            nn.Conv2d(256, 313, 1, 1, 0)            # Проекция в 313-мерное пространство
        )
        # 313 измерения соответствуют квантованному цветовому пространству ab-каналов
        # Каждое измерение - вероятность определенного цвета

        # ====================
        # ВЫХОДНОЙ СЛОЙ
        # Преобразование вероятностей в ab-каналы
        # ====================
        
        # Softmax: преобразование в вероятностное распределение
        self.softmax = nn.Softmax(dim=1)  # По dimension=1 (по каналам)
        
        # Линейный слой: 313 вероятностей → 2 ab-канала
        # Матрица перехода 313×2 обучена на наборе данных
        self.model_out = nn.Conv2d(313, 2, 1, 1, 0, bias=False)
        
        # Upsample 4x: окончательное увеличение до исходного размера
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        """
        ПРЯМОЙ ПРОХОД (FORWARD PASS)
        =============================
        Преобразует L-канал (яркость) в ab-каналы (цветность)
        
        Args:
            input_l: Тензор [batch_size, 1, H, W] - L-канал изображения
            
        Returns:
            Тензор [batch_size, 2, H, W] - предсказанные ab-каналы
        
        Алгоритм:
        1. Нормализация L-канала
        2. Проход через кодировщик (уменьшение размера)
        3. Проход через промежуточные слои
        4. Проход через декодировщик (увеличение размера)
        5. Преобразование в цветовое пространство
        6. Денормализация ab-каналов
        """
        # 1. Нормализация входного L-канала
        x = self.model1(self.normalize_l(input_l))
        
        # 2. Кодировщик: последовательное уменьшение размера
        x = self.model2(x)  # Downsample 2x → 1/4 от оригинала
        x = self.model3(x)  # Downsample 2x → 1/8 от оригинала
        x = self.model4(x)  # Без downsampling
        
        # 3. Промежуточные слои с увеличенным receptive field
        x = self.model5(x)  # Dilated convolutions
        x = self.model6(x)  # Dilated convolutions
        x = self.model7(x)  # Обычные свертки
        
        # 4. Декодировщик: upsampling и генерация цветности
        x = self.model8(x)  # Upsample 2x, выход 313 каналов
        
        # 5. Преобразование в цветовое пространство
        # Softmax: 313 каналов → вероятностное распределение
        # model_out: 313 вероятностей → 2 ab-канала
        out = self.model_out(self.softmax(x))
        
        # 6. Upsample 4x до исходного размера и денормализация
        return self.unnormalize_ab(self.upsample4(out))


def eccv16(pretrained=True):
    """
    ФАБРИЧНЫЙ МЕТОД ДЛЯ СОЗДАНИЯ ECCV16 МОДЕЛИ
    ==========================================
    
    Args:
        pretrained: Загружать предобученные веса (True/False)
                    Веса обучены на 1.3 млн изображений ImageNet
    
    Returns:
        ECCVGenerator: Инициализированная модель
        
    Особенности предобученной модели:
    - Обучена на 1.3 миллионах цветных изображений
    - Использует rebalancing классов для редких цветов
    - Выходное распределение соответствует естественным цветам
    """
    # Создание экземпляра модели
    model = ECCVGenerator()
    
    # Загрузка предобученных весов если требуется
    if pretrained:
        import torch.utils.model_zoo as model_zoo
        
        # URL весов модели (официальный релиз авторов)
        model_url = 'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth'
        
        # Загрузка state_dict с проверкой хеша для безопасности
        model.load_state_dict(
            model_zoo.load_url(
                model_url,
                map_location='cpu',      # Загрузка на CPU (можно изменить на 'cuda')
                check_hash=True          # Проверка целостности файла
            )
        )
        
        # Установка модели в режим оценки (не обучения)
        model.eval()
        
        print(f"✅ ECCV16 модель загружена с предобученными весами")
    
    return model