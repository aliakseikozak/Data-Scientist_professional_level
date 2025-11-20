import torch
import os
import sys
import yaml
import subprocess
import re
import pandas as pd
from IPython.display import display, clear_output

def setup_training_command(data_yaml_path):
    return [
        sys.executable, "yolov5/train.py",
        "--img", "416",
        "--batch", "4",
        "--epochs", "50",           # количество эпох
        "--data", data_yaml_path,
        "--weights", "yolov5s.pt",
        "--freeze", "10",
        "--noplots",
        "--project", "results",
        "--name", "football_model",
        "--exist-ok",
        "--save-period", "-1",
        "--patience", "15"
    ]

def check_data_yaml(data_yaml_path):
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Файл {data_yaml_path} не найден")
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"Количество классов: {len(data_config['names'])}")
    print(f"Названия классов: {data_config['names']}")

def install_requirements():
    req_file = "yolov5/requirements.txt"
    if os.path.exists(req_file):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], check=True)

def train_football_model(data_yaml_path="dataset/data.yaml"):
    os.environ["PYTHONWARNINGS"] = "ignore"

    check_data_yaml(data_yaml_path)
    install_requirements()
    cmd = setup_training_command(data_yaml_path)
    print("\n🚀 Запуск обучения...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    metrics_pattern = re.compile(
        r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )

    # Таблица для метрик
    df_metrics = pd.DataFrame(columns=["Эпоха", "P", "R", "mAP50", "mAP50-95"])
    epoch_counter = 0

    for line in process.stdout:
        line = line.strip()
        m_metrics = metrics_pattern.search(line)
        if m_metrics:
            epoch_counter += 1
            P, R, mAP50, mAP5095 = m_metrics.groups()
            df_metrics.loc[epoch_counter - 1] = [epoch_counter, P, R, mAP50, mAP5095]

            # Чистим вывод и показываем обновленную таблицу
            clear_output(wait=True)
            display(df_metrics)

    process.wait()
    if process.returncode == 0:
        print("\n🎉 Модель обучена! Результаты в results/football_model/weights/best.pt")
    else:
        print(f"\n❌ Ошибка при обучении, код выхода: {process.returncode}")

if __name__ == "__main__":
    train_football_model()
