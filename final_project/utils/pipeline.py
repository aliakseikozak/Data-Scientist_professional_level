"""
ПАЙПЛАЙН ОБРАБОТКИ ИЗОБРАЖЕНИЙ
===============================
Основная логика стилизации фотографий:
1. Определение ЧБ/цветного изображения
2. Предварительная раскраска ЧБ фото (если нужно)
3. Применение выбранного стиля
4. Постобработка и возврат результата

Используемые технологии:
- Pillow для базовой обработки
- NumPy для работы с массивами
- Собственные фильтры для стилей
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from utils.models import realistic_color, cartoon_color, modern_color

# Карта доступных стилей (синхронизирована с bot.py)
STYLE_MAP = [
    "Реалистично",
    "Холодные тона", 
    "Сепия",
    "Советские",
    "Дореволюционные",
    "90-е",
    "Современные",
    "Винтаж Европа/Америка",
    "Ретро (50-80)"
]

def apply_sepia(img: Image.Image, intensity=0.5) -> Image.Image:
    """
    Применение сепия-фильтра через цветовую матрицу
    Формула основана на классическом сепия преобразовании RGB
    """
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    # Матрица преобразования сепия
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
    
    sep = np.stack([tr, tg, tb], axis=2)
    sep = np.clip(sep*intensity + arr*(1-intensity), 0, 255)
    return Image.fromarray(sep.astype(np.uint8))

def apply_vignette(img: Image.Image) -> Image.Image:
    """
    Создание эффекта виньетирования (затемнение по краям)
    Имитирует старые объективы и добавляет винтажный вид
    """
    w, h = img.size
    mask = Image.new("L", (w,h))
    
    # Создание градиентной маски
    for y in range(h):
        for x in range(w):
            dx = (x - w/2)/(w/2)
            dy = (y - h/2)/(h/2)
            val = int(255 * (dx*dx + dy*dy))  # Квадратичное затухание
            mask.putpixel((x,y), min(val,255))
    
    img.putalpha(mask)
    return img.convert("RGB")

def is_grayscale_tolerant(img: Image.Image, threshold=0.07) -> bool:
    """
    УЛУЧШЕННАЯ ПРОВЕРКА ЧЕРНО-БЕЛОГО ИЗОБРАЖЕНИЯ
    ============================================
    Толерантная проверка с учетом:
    - старых/состаренных фото с легким оттенком
    - минимального цветового шума от сканирования
    - небольших цветовых вариаций
    
    Args:
        img: PIL Image
        threshold: максимальная допустимая разница между каналами (0-1)
                  0.07 = 7% разницы, оптимально для старых фото
    
    Returns:
        True если изображение можно считать черно-белым
    """
    img_np = np.array(img)
    
    # Случай 1: уже 2D массив (один канал) - точно ЧБ
    if img_np.ndim == 2:
        return True
    
    # Случай 2: 3D массив (RGB или RGBA)
    elif img_np.ndim == 3:
        # Если RGBA, берем только RGB каналы
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]
        
        if img_np.shape[2] == 3:
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            
            # Вычисляем среднюю разницу между каналами
            diff_rg = np.abs(r - g).mean() / 255.0
            diff_gb = np.abs(g - b).mean() / 255.0
            diff_br = np.abs(b - r).mean() / 255.0
            
            # Максимальная разница между любыми двумя каналами
            max_diff = max(diff_rg, diff_gb, diff_br)
            
            # Дополнительная проверка: если изображение очень темное/светлое
            avg_brightness = img_np.mean() / 255.0
            
            # Логи для отладки (можно удалить после тестирования)
            # print(f"Max channel difference: {max_diff:.3f}")
            # print(f"Average brightness: {avg_brightness:.3f}")
            
            # Если разница меньше порога, считаем ЧБ
            return max_diff < threshold
    
    return False

def normalize_grayscale_image(img: Image.Image) -> Image.Image:
    """
    Нормализация потенциально ЧБ изображений
    Убирает легкие цветовые оттенки, делая фото чисто черно-белым
    """
    # Если уже в режиме 'L' (grayscale)
    if img.mode == 'L':
        return img.convert('RGB')
    
    # Конвертируем в градации серого и обратно в RGB
    gray = img.convert('L')
    return gray.convert('RGB')

def is_grayscale(img: Image.Image) -> bool:
    """
    СТАРАЯ ВЕРСИЯ (оставлена для обратной совместимости)
    Использует новую толерантную проверку
    """
    return is_grayscale_tolerant(img, threshold=0.07)

def process(img_np: np.ndarray, style: str) -> np.ndarray:
    """
    ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
    ===========================
    Args:
        img_np: изображение в формате numpy array [0, 1]
        style: название стиля из STYLE_MAP
    
    Returns:
        np.ndarray: обработанное изображение [0, 1]
    
    Алгоритм:
    1. Конвертация в PIL Image
    2. Определение ЧБ/цветное (УЛУЧШЕННАЯ ПРОВЕРКА)
    3. Предобработка (раскраска ЧБ если нужно)
    4. Применение стиля
    5. Возврат результата
    """
    # Конвертация в PIL Image
    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
    
    # УЛУЧШЕННАЯ ПРОВЕРКА на черно-белое
    is_gray = is_grayscale_tolerant(pil_img, threshold=0.07)
    
    # Дополнительно: если определили как ЧБ, нормализуем изображение
    if is_gray:
        pil_img = normalize_grayscale_image(pil_img)
    
    # Стили, которые требуют предварительной раскраски ЧБ фото
    styles_needing_color = ["Советские", "90-е", "Ретро (50-80)"]
    
    # Если фото ЧБ и стиль требует раскраски
    if is_gray and style in styles_needing_color:
        img_to_process = modern_color(pil_img)  # Предварительная раскраска
    else:
        img_to_process = pil_img
    
    # ОБРАБОТКА ПО СТИЛЯМ
    # ====================
    
    if style == "Реалистично":
        # Реалистичное раскрашивание только для ЧБ фото
        out = realistic_color(pil_img) if is_gray else pil_img
        
    elif style == "Холодные тона":
        # Холодная цветовая гамма с усилением контраста
        base_img = modern_color(pil_img) if is_gray else pil_img
        out = cartoon_color(base_img)  # Базовая стилизация
        
        # Дополнительная настройка
        out = ImageEnhance.Brightness(out).enhance(1.1)    # +10% яркости
        out = ImageEnhance.Contrast(out).enhance(1.2)      # +20% контраста
        out = ImageEnhance.Color(out).enhance(1.2)         # +20% насыщенности
        
    elif style == "Сепия":
        # Классический сепия-фильтр
        out = apply_sepia(pil_img, intensity=0.8)
        
    elif style == "Советские":
        # Стиль старых советских фото
        out = apply_sepia(img_to_process, intensity=0.5)
        out = ImageEnhance.Color(out).enhance(0.8)     # -20% насыщенности
        out = ImageEnhance.Contrast(out).enhance(1.1)  # +10% контраста
        
    elif style == "Дореволюционные":
        # Эффект очень старых фото
        out = apply_sepia(pil_img, intensity=1.0)      # Полная сепия
        out = out.filter(ImageFilter.GaussianBlur(radius=1))  # Легкое размытие
        
    elif style == "90-е":
        # Яркие, насыщенные цвета 90-х
        out = ImageEnhance.Color(img_to_process).enhance(1.4)  # +40% насыщенности
        out = out.filter(ImageFilter.GaussianBlur(radius=0.8)) # Легкое размытие
        
    elif style == "Современные":
        # Современное раскрашивание только для ЧБ
        out = modern_color(pil_img) if is_gray else pil_img
        
    elif style == "Винтаж Европа/Америка":
        # Винтажный западный стиль
        out = apply_sepia(pil_img, intensity=0.3)      # Легкая сепия
        out = out.filter(ImageFilter.GaussianBlur(radius=0.5)) # Слабое размытие
        
    elif style == "Ретро (50-80)":
        # Стиль середины 20 века
        out = apply_sepia(img_to_process, intensity=0.3)
        out = ImageEnhance.Contrast(out).enhance(1.2)  # +20% контраста
        out = apply_vignette(out)                      # Виньетирование
        
    else:
        # На всякий случай возвращаем оригинал
        out = pil_img

    # Конвертация обратно в numpy array [0, 1]
    return np.array(out).astype(np.float32) / 255.0