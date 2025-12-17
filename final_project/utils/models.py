"""
ИНТЕРФЕЙСЫ К НЕЙРОСЕТЕВЫМ МОДЕЛЯМ
==================================
Обеспечивает взаимодействие с предобученными моделями раскрашивания:
1. ECCV16 - быстрая модель для стилизации
2. SIGGRAPH17 - продвинутая модель для реалистичного раскрашивания

Особенности:
- Автоматическая загрузка весов
- Конвертация между цветовыми пространствами (RGB ↔ LAB)
- Ресайз изображений под требования моделей
"""

import torch
import numpy as np
from PIL import Image
from skimage import color  # Для работы с LAB цветовым пространством
import cv2
import torch.nn.functional as F

# Импорт моделей из colorizers
from colorizers.eccv16 import eccv16
from colorizers.siggraph17 import siggraph17

# Устройство для вычислений (CPU/GPU)
device = "cpu"

# Загрузка предобученных моделей
ECCV16_MODEL = eccv16(pretrained=True).eval().to(device)
SIGGRAPH_MODEL = siggraph17()
SIGGRAPH_MODEL.eval()

# ============================
# СТИЛЬ "CARTOON" (ECCV16)
# ============================
def cartoon_color(pil_img):
    """
    Стилизация в "мультяшном" стиле с помощью ECCV16
    Особенности:
    - Быстрая обработка
    - Усиленные цвета
    - Художественный вид
    """
    # Конвертация в grayscale (L-канал)
    L_np = np.array(pil_img.convert("L")).astype(np.float32)
    L_tensor = torch.tensor(L_np, dtype=torch.float32)[None, None, :, :]

    # Прямой проход через модель
    ab_out = ECCV16_MODEL(L_tensor)

    # Ресайз AB-каналов к оригинальному размеру
    ab_out_orig = F.interpolate(ab_out, size=L_tensor.shape[2:], mode='bilinear')

    # Объединение L + AB → LAB
    lab_out = torch.cat((L_tensor, ab_out_orig), dim=1)
    
    # Конвертация LAB → RGB
    rgb_out = color.lab2rgb(lab_out[0].detach().permute(1, 2, 0).cpu().numpy())

    # Конвертация в PIL Image
    rgb_out = (rgb_out * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_out)

# ============================
# СТИЛЬ "REALISTIC" (SIGGRAPH17)
# ============================
def realistic_color(pil_img: Image.Image) -> Image.Image:
    """
    Реалистичное раскрашивание черно-белых фото
    Особенности:
    - Фотореалистичный результат
    - Естественные цвета
    - Сохранение деталей
    
    Использует SIGGRAPH17 - state-of-the-art модель раскрашивания
    """
    img_np = np.array(pil_img)
    H, W = img_np.shape[:2]

    # 1. Оригинальный L-канал (яркость)
    img_lab_orig = color.rgb2lab(img_np)
    l_orig = img_lab_orig[:, :, 0]
    tens_l_orig = torch.tensor(l_orig, dtype=torch.float32)[None, None, :, :]

    # 2. Ресайз L-канала до 256x256 для модели
    img_lab_rs = color.rgb2lab(np.array(pil_img.resize((256, 256), Image.BICUBIC)))
    l_rs = img_lab_rs[:,:,0]
    tens_l_rs = torch.tensor(l_rs, dtype=torch.float32)[None, None, :, :]

    # 3. Прямой проход через SIGGRAPH17 модель
    with torch.no_grad():
        out_ab = SIGGRAPH_MODEL(tens_l_rs)  # [1, 2, H, W]

    # 4. Ресайз AB-каналов обратно к оригинальному размеру
    out_ab_orig = F.interpolate(out_ab, size=(H, W), mode='bilinear')

    # 5. Объединение L + AB → LAB → RGB
    lab_out = torch.cat((tens_l_orig, out_ab_orig), dim=1)
    rgb_out = color.lab2rgb(lab_out[0].permute(1, 2, 0).cpu().numpy())

    return Image.fromarray((rgb_out * 255).astype(np.uint8))

# ============================
# СТИЛЬ "MODERN" (ECCV16 - CANONICAL)
# ============================
def modern_color(pil_img: Image.Image) -> Image.Image:
    """
    Современное раскрашивание по каноническому алгоритму ECCV16
    
    Алгоритм:
    1. RGB → LAB
    2. Извлечение L-канала
    3. Ресайз до 256x256
    4. Прямой проход через ECCV16
    5. Ресайз AB-каналов
    6. LAB → RGB
    """
    img_rgb = np.array(pil_img)

    # 1. Оригинальный L-канал
    img_lab_orig = color.rgb2lab(img_rgb)
    L_orig = img_lab_orig[:, :, 0]
    tens_l_orig = torch.tensor(L_orig, dtype=torch.float32)[None, None, :, :]

    # 2. Ресайз RGB → 256x256 (используем OpenCV для совместимости)
    img_rs = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_lab_rs = color.rgb2lab(img_rs)
    L_rs = img_lab_rs[:, :, 0]
    tens_l_rs = torch.tensor(L_rs, dtype=torch.float32)[None, None, :, :]

    # 3. Прямой проход через ECCV16
    with torch.no_grad():
        out_ab = ECCV16_MODEL(tens_l_rs)

    # 4. Постобработка (как в оригинальном demo_release.py)
    out_ab_orig = F.interpolate(out_ab, size=tens_l_orig.shape[2:], mode='bilinear')
    lab_out = torch.cat((tens_l_orig, out_ab_orig), dim=1)

    # 5. Конвертация LAB → RGB
    rgb_out = color.lab2rgb(
        lab_out[0].permute(1, 2, 0).cpu().numpy()
    )

    # 6. Возврат PIL Image
    return Image.fromarray(
        (rgb_out * 255).clip(0, 255).astype(np.uint8)
    )