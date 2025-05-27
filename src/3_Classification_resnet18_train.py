import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 基本設定 ==========
image_size = 224
batch_size = 32
num_classes = 3
num_epochs = 20
learning_rate = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的設備: {device}")

# ========== 資料前處理 ==========
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# ========== 資料集載入 ==========
train_dir = 'data/dentistry_split/train'
val_dir = 'data/dentistry_split/val'
test_dir = 'data/dentistry_split/test'

# 資料夾結構檢查
print("\n📁 檢查資料夾內容:")
for subset, path in [('Train', train_dir), ('Val', val_dir), ('Test', test_dir)]:
    print(f"\n🔍 {subset} 資料夾：{path}")
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            image_count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            print(f"  - {class_name}: {image_count} 張圖片")
        else:
            print(f"⚠️ 非資料夾項目：{class_name}")

# 載入資料集
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
print(f"\n📌 分類標籤：{class_names}")
print(f"✅ 訓練樣本數：{len(train_dataset)}")
print(f"✅ 驗證樣本數：{len(val_dataset)}")

# ========== 建立模型 ==========
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 避免警告
for param in resnet.parameters():
    param.requires_grad = False  # 凍結特徵萃取層
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model = resnet.to(device)

# ========== 損失與優化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# ========== Early Stopping ==========
early_stopping_patience = 3
best_val_loss = float('inf')
epochs_without_improvement = 0

# ========== 訓練開始 ==========
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

    # 驗證
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
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
    print(f"📊 驗證結果 | Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_resnet18_model.pth")
        print(f"✅ 儲存最佳模型 @ epoch {epoch+1}")
    else:
        epochs_without_improvement += 1
        print(f"⚠️ 驗證損失未改善 {epochs_without_improvement} 次")

    scheduler.step()
    if epochs_without_improvement >= early_stopping_patience:
        print("⏹️ 觸發 Early Stopping！")
        break

print("\n✅ 訓練完成！")

# ========== 評估函數 ==========
def evaluate_model(loader, name="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n📌 {name} 評估報告：")
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1 Score (weighted): {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    conf_mat = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}_confusion_matrix_resnet18.png")
    plt.show()

    print("\n📊 每類別準確率：")
    for i, class_name in enumerate(class_names):
        correct = conf_mat[i, i]
        total = conf_mat[i].sum()
        acc = correct / total if total > 0 else 0
        print(f"{class_name}: {acc:.4f} ({correct}/{total})")

# ========== 載入最佳模型並進行評估 ==========
model.load_state_dict(torch.load("best_resnet18_model.pth"))
evaluate_model(val_loader, name="Validation")
evaluate_model(test_loader, name="Test")
