import os

def pad_num(num, width=4):
    """Добавляем ведущие нули к номеру изображения, например 2 → 0002"""
    return str(num).zfill(width)

def csv_to_pairs_txt(pairs_file, out_file):
    """
    Конвертирует pairsDevTrain / pairsDevTest в единый текстовый файл для LFWDataset
    """
    lines = []

    with open(pairs_file, 'r') as f:
        all_lines = f.readlines()

    # Пропускаем первую строку, если там количество пар
    start_idx = 0
    if all_lines[0].strip().isdigit():
        start_idx = 1

    for line in all_lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) == 3:
            # Положительная пара: name, img1, img2
            name = parts[0]
            img1 = pad_num(parts[1])
            img2 = pad_num(parts[2])
            label = 1
            lines.append(f"{name} {img1} {img2} {name} {img1} {img2} {label}\n")
        elif len(parts) == 4:
            # Отрицательная пара: name1, img1, name2, img2
            name1 = parts[0]
            img1 = pad_num(parts[1])
            name2 = parts[2]
            img2 = pad_num(parts[3])
            label = 0
            lines.append(f"{name1} {img1} {img1} {name2} {img2} {img2} {label}\n")
        else:
            # Пропускаем пустые или некорректные строки
            continue

    # Сохраняем в файл
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        f.writelines(lines)

    print(f"✅ Создан файл {out_file} с {len(lines)} парами")


if __name__ == "__main__":
    # Пример запуска
    csv_to_pairs_txt('data/pairsDevTrain.txt', 'data/pairs_train.txt')
    csv_to_pairs_txt('data/pairsDevTest.txt', 'data/pairs_test.txt')
