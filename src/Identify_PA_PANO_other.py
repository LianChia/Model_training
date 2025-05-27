import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import shutil

# 模型架構（需與訓練時一致）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (128 // 8) * (128 // 8), 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 3
class_names = ['PA', 'PANO', 'other']  # 根據你的資料夾名稱設定

# 圖像轉換（與訓練時相同）
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 載入模型
model = SimpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('classification_model.pth', map_location=device))
model.eval()

# 測試圖片資料夾 (假設所有圖片都在 `test2` 資料夾)
test2_dir = './data/test'  # 放測試圖片的資料夾
test_images = os.listdir(test2_dir)

# 創建PA、PANO、other子資料夾
for class_name in class_names:
    os.makedirs(os.path.join(test2_dir, class_name), exist_ok=True)

# 設定信心值閾值
threshold = 0.8  # 可根據驗證集調整

# 預測並分類圖片
for img_name in test_images:
    img_path = os.path.join(test2_dir, img_name)
    if os.path.isfile(img_path):  # 確保是檔案而非資料夾
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 計算每個類別的概率
            confidence, predicted = torch.max(probs, 1)  # 最大概率和對應的類別
            confidence_value = confidence.item()  # 獲取信心值（概率）

            # 如果信心值低於閾值，直接分類為 other
            if confidence_value < threshold:
                predicted_class = 'other'
            else:
                predicted_class = class_names[predicted.item()]

        # 打印預測結果和信心值
        print(f"{img_name} ➜ 預測為：{predicted_class} (信心值: {confidence_value:.4f})")

        # 將圖片移動到對應的資料夾
        target_folder = os.path.join(test2_dir, predicted_class)
        shutil.move(img_path, os.path.join(target_folder, img_name))  # 移動圖片到相應的資料夾

print("所有圖片已分類並移動。")
