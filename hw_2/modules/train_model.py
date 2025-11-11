import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score

# ===============================================
# 1️⃣ Dataset
# ===============================================
class LFWDataset(Dataset):
    """Класс для загрузки пар изображений из LFW"""
    def __init__(self, pairs_txt, root_dir, transform=None):
        self.pairs = []
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(pairs_txt):
            raise FileNotFoundError(f"{pairs_txt} не найден!")

        with open(pairs_txt, 'r') as f:
            next(f)  # пропускаем заголовок, если есть
            for line in f:
                parts = line.strip().split('\t')  # используем табуляцию, как в рабочем ноутбуке
                if len(parts) == 3:  # matchpair
                    self.pairs.append({
                        'img1_name': parts[0], 'img1_num': int(parts[1]),
                        'img2_name': parts[0], 'img2_num': int(parts[2]),
                        'label': 1
                    })
                elif len(parts) == 4:  # mismatchpair
                    self.pairs.append({
                        'img1_name': parts[0], 'img1_num': int(parts[1]),
                        'img2_name': parts[2], 'img2_num': int(parts[3]),
                        'label': 0
                    })

        if len(self.pairs) == 0:
            raise ValueError(f"Датасет пуст! Проверьте файл {pairs_txt} и путь {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1_path = os.path.join(self.root_dir, pair['img1_name'], f"{pair['img1_name']}_{pair['img1_num']:04d}.jpg")
        img2_path = os.path.join(self.root_dir, pair['img2_name'], f"{pair['img2_name']}_{pair['img2_num']:04d}.jpg")

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(pair['label'], dtype=torch.float32)
        return img1, img2, label


# ===============================================
# 2️⃣ Сиамская сеть на ResNet18
# ===============================================
class SiameseResNet(nn.Module):
    """ResNet18 без классификатора + L2-нормированный эмбеддинг"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.fc = nn.Linear(512, 256)

    def forward_one(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ===============================================
# 3️⃣ Контрастная функция потерь
# ===============================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = torch.norm(out1 - out2, p=2, dim=1)
        loss = label * dist**2 + (1 - label) * torch.clamp(self.margin - dist, min=0)**2
        return loss.mean(), dist


# ===============================================
# 4️⃣ Валидация
# ===============================================
def validate(model, dataloader, device, margin=1.0):
    model.eval()
    all_labels, all_dists = [], []
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            dist = torch.norm(out1 - out2, p=2, dim=1)
            all_labels.append(label)
            all_dists.append(dist)

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_dists = torch.cat(all_dists).cpu().numpy()

    pred = (all_dists < margin).astype(int)
    accuracy = (pred == all_labels).mean()
    pos_mean = all_dists[all_labels == 1].mean() if (all_labels == 1).any() else 0
    neg_mean = all_dists[all_labels == 0].mean() if (all_labels == 0).any() else 0
    try:
        auc = roc_auc_score(all_labels, -all_dists)
    except:
        auc = 0.0

    return accuracy, pos_mean, neg_mean, auc


# ===============================================
# 5️⃣ Главная функция обучения
# ===============================================
def train_model(
    train_pairs,
    test_pairs,
    root_dir,
    batch_size=16,
    lr=1e-4,
    margin=1.5,
    epochs=50,
    patience=5,
    save_path="models/siamese_resnet_best.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor()
    ])

    train_dataset = LFWDataset(train_pairs, root_dir, transform=transform)
    test_dataset = LFWDataset(test_pairs, root_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SiameseResNet(pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss, _ = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc, pos_mean, neg_mean, auc = validate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {acc:.4f} | PosDist: {pos_mean:.4f} | NegDist: {neg_mean:.4f} | AUC: {auc:.4f}")

        # Early Stopping
        if auc > best_auc:
            best_auc = auc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"🔹 Early stopping на эпохе {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Обучение завершено. Лучшая модель сохранена в {save_path} (AUC={best_auc:.4f})")

    return model, best_auc
