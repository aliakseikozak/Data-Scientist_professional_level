"""
МОДЕЛЬ SIGGRAPH17 ДЛЯ РЕАЛИСТИЧНОГО РАСКРАШИВАНИЯ ИЗОБРАЖЕНИЙ
=============================================================
Модель из статьи "Real-Time User-Guided Image Colorization with Learned Deep Priors" 
(Zhang et al., SIGGRAPH 2017)

УСОВЕРШЕНСТВОВАННАЯ АРХИТЕКТУРА ПО СРАВНЕНИЮ С ECCV16:
1. Поддержка пользовательских подсказок (цветные мазки)
2. Многоуровневые пропускающие соединения (skip connections)
3. Двойной выход: классификация + регрессия
4. Более глубокая архитектура с лучшим качеством

Ключевые инновации:
- Интерактивное раскрашивание с пользовательскими подсказками
- Сочетание классификации и регрессии для точности цвета
- Пирамидальная архитектура с пропускающими соединениями
"""

import torch
import torch.nn as nn

from .base_color import BaseColor  # Базовый класс для работы с LAB цветовым пространством


class SIGGRAPHGenerator(BaseColor):
    """
    ГЕНЕРАТОР SIGGRAPH17
    ====================
    Продвинутая модель для фотореалистичного раскрашивания с поддержкой
    пользовательских подсказок (цветных мазков).
    
    Особенности архитектуры:
    - Вход: L-канал + опциональные ab-подсказки + маска подсказок
    - Пирамидальная структура с пропускающими соединениями
    - Двойной выход (классификация + регрессия) для точности
    - Адаптивный upsampling с сохранением деталей
    """
    
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        """
        ИНИЦИАЛИЗАЦИЯ АРХИТЕКТУРЫ SIGGRAPH17
        
        Args:
            norm_layer: Слой нормализации (BatchNorm2d)
            classes: Количество классов в классификационном выходе (529)
            
        Входные каналы: 4 = [L-канал] + [a-канал подсказки] + [b-канал подсказки] + [маска подсказки]
        Выход: 2 канала (ab) для регрессии + classes каналов для классификации
        """
        super(SIGGRAPHGenerator, self).__init__()

        # ====================
        # ЭТАП 1: КОДИРОВЩИК С ПИРАМИДАЛЬНЫМ ДОУНСЭМПЛИНГОМ
        # Каждый блок уменьшает разрешение в 2 раза (stride=2 через сэмплирование)
        # ====================
        
        # Conv1: 4 входных канала → 64 канала
        # Вход: [L, a_hints, b_hints, mask]
        model1=[
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),  # Сохраняет разрешение
            nn.ReLU(True),                                                    # Нелинейность
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), # Углубление признаков
            nn.ReLU(True),
            norm_layer(64),                                                   # Нормализация
        ]
        # После conv1: ручной downsampling 2x [::2, ::2] в forward

        # Conv2: 64 → 128 каналов
        model2=[
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]
        # После conv2: downsampling 2x

        # Conv3: 128 → 256 каналов (3 сверточных слоя)
        model3=[
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),  # Дополнительная емкость
            nn.ReLU(True),
            norm_layer(256),
        ]
        # После conv3: downsampling 2x → всего 8x уменьшение

        # Conv4: 256 → 512 каналов (базовый блок)
        model4=[
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        # Нет downsampling - минимальное разрешение достигнуто

        # ====================
        # ЭТАП 2: ПРОМЕЖУТОЧНЫЕ СЛОИ С DILATED CONVOLUTIONS
        # Увеличение receptive field при сохранении разрешения
        # ====================
        
        # Conv5-6: Dilated convolutions (dilation=2)
        # Увеличивают область восприятия без потери информации
        model5=[
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),  # padding=2 для сохранения размера
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]
        
        model6=[
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Conv7: Возврат к обычным сверткам для уточнения
        model7=[
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # ====================
        # ЭТАП 3: ДЕКОДИРОВЩИК С ПРОПУСКАЮЩИМИ СОЕДИНЕНИЯМИ (SKIP CONNECTIONS)
        # U-Net like архитектура: соединение высоко- и низкоуровневых признаков
        # ====================
        
        # Conv8: Upsampling 2x + skip connection из conv3
        model8up=[
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)  # Transposed conv для upsampling
        ]
        model3short8=[
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),  # Проекция признаков conv3
        ]
        model8=[
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),  # Уточнение после объединения
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]
        # Формула: conv8 = model8(model8up(conv7) + model3short8(conv3))

        # Conv9: Upsampling 2x + skip connection из conv2
        model9up=[
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        model2short9=[
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),  # Проекция признаков conv2
        ]
        model9=[
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]
        # conv9 = model9(model9up(conv8) + model2short9(conv2))

        # Conv10: Upsampling 2x + skip connection из conv1
        model10up=[
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
        ]
        model1short10=[
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),  # Проекция признаков conv1
        ]
        model10=[
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=.2),  # LeakyReLU для улучшения градиентов
        ]
        # conv10 = model10(model10up(conv9) + model1short10(conv1))

        # ====================
        # ВЫХОДНЫЕ СЛОИ: ДВОЙНОЙ ВЫХОД
        # 1. Классификация: 529 классов (точное определение цвета)
        # 2. Регрессия: 2 канала ab (непрерывное предсказание)
        # ====================
        
        # Выход классификации: 256 каналов → 529 классов
        model_class=[
            nn.Conv2d(256, classes, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
        ]
        # Используется для точного предсказания дискретных цветов

        # Выход регрессии: 128 каналов → 2 ab канала
        model_out=[
            nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=True),
            nn.Tanh()  # Ограничение выхода в диапазон [-1, 1]
        ]
        # Используется для плавного непрерывного предсказания

        # ====================
        # ИНИЦИАЛИЗАЦИЯ ВСЕХ СЛОЕВ
        # ====================
        
        # Кодировщик
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        
        # Промежуточные слои
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        
        # Декодировщик с skip connections
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        
        # Проекционные слои для skip connections
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        
        # Выходные слои
        self.model_class = nn.Sequential(*model_class)  # Классификация
        self.model_out = nn.Sequential(*model_out)      # Регрессия
        
        # Дополнительные слои
        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='bilinear'),])  # Финальный upsampling
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])  # Для классификационного выхода

    def forward(self, input_A, input_B=None, mask_B=None):
        """
        ПРЯМОЙ ПРОХОД С ПОДДЕРЖКОЙ ПОЛЬЗОВАТЕЛЬСКИХ ПОДСКАЗОК
        ======================================================
        
        Args:
            input_A: Тензор [batch, 1, H, W] - L-канал (яркость)
            input_B: Тензор [batch, 2, H, W] - пользовательские ab-подсказки (опционально)
            mask_B:  Тензор [batch, 1, H, W] - маска подсказок (где применять input_B)
            
        Returns:
            Тензор [batch, 2, H, W] - предсказанные ab-каналы
        
        Особенности:
        - Если input_B=None, используется нулевые подсказки
        - Маска определяет, где применять пользовательские цвета
        - Skip connections сохраняют детали на разных уровнях
        """
        
        # Инициализация подсказок если не предоставлены
        if input_B is None:
            # Нулевые ab-каналы: нет пользовательских подсказок
            input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
        
        if mask_B is None:
            # Нулевая маска: применяем ко всему изображению
            mask_B = input_A * 0
        
        # ====================
        # ЭТАП 1: ПОДГОТОВКА ВХОДА
        # Конкатенация: [нормализованный L, нормализованные ab, маска]
        # Всего 4 канала: 1(L) + 2(ab) + 1(mask) = 4
        # ====================
        input_concat = torch.cat(
            (self.normalize_l(input_A), self.normalize_ab(input_B), mask_B),
            dim=1
        )
        
        # ====================
        # ЭТАП 2: КОДИРОВЩИК С ПИРАМИДАЛЬНЫМ ДОУНСЭМПЛИНГОМ
        # Ручной downsampling через сэмплирование [::2, ::2]
        # ====================
        
        # Conv1: полное разрешение → признаки низкого уровня
        conv1_2 = self.model1(input_concat)
        
        # Conv2: downsampling 2x → признаки среднего уровня
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])  # Выбор каждого 2-го пикселя
        
        # Conv3: downsampling 4x от оригинала → признаки высокого уровня
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        
        # Conv4: downsampling 8x → глубокие абстрактные признаки
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        
        # ====================
        # ЭТАП 3: ПРОМЕЖУТОЧНЫЕ СЛОИ
        # Работа на низком разрешении с увеличенным receptive field
        # ====================
        conv5_3 = self.model5(conv4_3)      # Dilated convolutions
        conv6_3 = self.model6(conv5_3)      # Dilated convolutions
        conv7_3 = self.model7(conv6_3)      # Обычные свертки
        
        # ====================
        # ЭТАП 4: ДЕКОДИРОВЩИК С SKIP CONNECTIONS (U-Net стиль)
        # Восстановление разрешения с сохранением деталей
        # ====================
        
        # Conv8: Upsample 2x + skip connection из conv3
        # model8up: 512→256, увеличение в 2 раза
        # model3short8: проекция conv3 (256→256) для согласования размеров
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)  # Уточнение объединенных признаков
        
        # Conv9: Upsample 2x + skip connection из conv2
        # Восстановление до 1/4 от оригинального разрешения
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        
        # Conv10: Upsample 2x + skip connection из conv1
        # Восстановление до полного разрешения
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        
        # ====================
        # ЭТАП 5: ВЫХОД РЕГРЕССИИ
        # Предсказание непрерывных ab-каналов
        # ====================
        out_reg = self.model_out(conv10_2)
        
        # ====================
        # ЭТАП 7: ДЕНОРМАЛИЗАЦИЯ И ВОЗВРАТ
        # ====================
        return self.unnormalize_ab(out_reg)


def siggraph17(pretrained=True):
    """
    ФАБРИЧНЫЙ МЕТОД ДЛЯ СОЗДАНИЯ SIGGRAPH17 МОДЕЛИ
    ==============================================
    
    Args:
        pretrained: Загружать предобученные веса (рекомендуется True)
                    Веса обучены на 1.3M изображениях с пользовательскими подсказками
    
    Returns:
        SIGGRAPHGenerator: Инициализированная модель
        
    Особенности предобученной модели:
    - Обучена с имитацией пользовательских подсказок (цветных мазков)
    - Multi-task learning: классификация + регрессия
    - Оптимизирована для фотореалистичных результатов
    - Поддерживает интерактивное раскрашивание
    """
    # Создание экземпляра модели
    model = SIGGRAPHGenerator()
    
    # Загрузка предобученных весов если требуется
    if pretrained:
        import torch.utils.model_zoo as model_zoo
        
        # URL официальных весов от авторов
        model_url = 'https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth'
        
        # Загрузка state_dict с проверкой целостности
        model.load_state_dict(
            model_zoo.load_url(
                model_url,
                map_location='cpu',      # Для CPU (можно изменить на 'cuda:0' для GPU)
                check_hash=True          # Проверка хеша файла
            )
        )
        
        # Установка модели в режим оценки (не обучения)
        model.eval()
        
        print(f"✅ SIGGRAPH17 модель загружена с предобученными весами")
    
    return model
