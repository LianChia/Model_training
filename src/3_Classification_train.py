import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 基本參數設定 ==========
image_size = 128
batch_size = 32
num_classes = 3
num_epochs = 20
learning_rate = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的設備: {device}")

# ========== 資料前處理 ==========
# 建議加上 Normalize，與推論一致
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ========== 資料集載入 ==========
train_dir = 'data/dentistry_split/train'
val_dir = 'data/dentistry_split/val'
test_dir = 'data/dentistry_split/test'
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = train_dataset.classes
print(f"分類標籤：{class_names}")

# ========== CNN 模型定義 ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (image_size // 8) * (image_size // 8), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ========== 初始化模型與訓練參數 ==========
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
early_stopping_patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

# ========== 訓練模型 ==========
for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch+1}/{num_epochs} 開始訓練...")

    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
            print(f"  🟢 批次 {i+1}/{len(train_loader)}")

    train_acc = correct / total
    avg_train_loss = running_loss / len(train_loader)
    print(f"✅ 訓練完成 | Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # ========== 驗證 ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # ========== Early Stopping ==========
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "classification_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1}")
    else:
        epochs_without_improvement += 1
        print(f"Validation loss did not improve for {epochs_without_improvement} epoch(s).")

    scheduler.step()
    if epochs_without_improvement >= early_stopping_patience:
        print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
        break

print("✅ Training complete.")

# ========== 評估函數 ==========
def evaluate_model(loader, dataset_name="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n📌 Final Evaluation on {dataset_name} Set")
    print(f"F1 Score (weighted): {val_f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
    conf_mat = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", conf_mat)
    # 🔳 Confusion Matrix 視覺化
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower()}_confusion_matrix.png")
    plt.show()
    print("\n📊 Per-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = conf_mat[i, i]
        class_total = conf_mat[i].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name}: {class_acc:.4f} ({class_correct}/{class_total})")

# ========== 載入最佳模型 ==========
model.load_state_dict(torch.load("classification_model.pth"))

# ========== 評估 Validation Set ==========
evaluate_model(val_loader, dataset_name="Validation")

# ========== 評估 Test Set ==========
evaluate_model(test_loader, dataset_name="Test")
