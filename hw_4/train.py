import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet
from config import Config
import logging
import matplotlib.pyplot as plt  # ✅ для графика

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def train_model():
    # ---------- Загрузка интентов ----------
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend([stem(word) for word in w])
            xy.append((w, tag))

    # Убираем дубликаты и сортируем
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # ---------- Создание обучающего датасета ----------
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words([stem(w) for w in pattern_sentence], all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=0)

    # ---------- Модель ----------
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64]
    output_size = len(tags)

    device = Config.DEVICE
    model = NeuralNet(input_size, hidden_sizes, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    epochs = 1000

    loss_history = []  # ✅ список для хранения loss по эпохам

    logger.info("=== НАЧАЛО ОБУЧЕНИЯ ===")
    for epoch in range(epochs):
        running_loss = 0.0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    logger.info("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

    # ---------- Сохранение модели ----------
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }, "data.pth")
    logger.info("Модель сохранена в data.pth")

    # ---------- Построение графика ----------
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs+1), loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Динамика loss во время обучения')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()
    logger.info("График loss сохранён как loss_plot.png")

if __name__ == "__main__":
    train_model()
