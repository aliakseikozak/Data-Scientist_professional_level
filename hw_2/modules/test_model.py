import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


# ============================================================
# 1️⃣ Модель (та же архитектура, что при обучении)
# ============================================================
class SiameseResNet(torch.nn.Module):
    """ResNet18 без классификатора + L2-нормированный эмбеддинг"""
    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        resnet.fc = torch.nn.Identity()
        self.backbone = resnet
        self.fc = torch.nn.Linear(512, 256)

    def forward_one(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=1)
        return out

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# -------------------------------
# 3️⃣ Функция сравнения лиц
# -------------------------------
def compare_faces(model_path, img_path1, img_path2, threshold=1.0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    model = SiameseResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загружаем изображения
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")

    t1 = transform(img1).unsqueeze(0).to(device)
    t2 = transform(img2).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        out1, out2 = model(t1, t2)
        dist = torch.norm(out1 - out2, p=2).item()

    # Схожесть в процентах
    similarity = max(0, 100 * (1 - dist / threshold))
    same_person = similarity >= 50

    label = f"{'✅ Один и тот же человек' if same_person else '❌ Разные люди'}\n" \
            f"Схожесть: {similarity:.1f}% | L2 Distance: {dist:.4f}"

    print(label)

    # Визуализация
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[0].set_title("Image 1")

    axs[1].imshow(img2)
    axs[1].axis("off")
    axs[1].set_title("Image 2")

    plt.suptitle(label, fontsize=12, color="green" if same_person else "red")
    plt.tight_layout()
    plt.show()

    return {
        "same_person": same_person,
        "similarity": similarity,
        "distance": dist
    }
