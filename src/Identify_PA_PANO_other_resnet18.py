import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 基本設定 ==========
image_size = 224
num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ["PA", "PANO", "other"]
confidence_threshold = 0.9  # <<<< 設定 softmax 閾值
print(f"使用的設備: {device}")

# ========== 圖片轉換 ==========
val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# ========== 載入模型 ==========
def load_model(weight_path='best_resnet18_model.pth'):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ========== 單張圖片預測（含閾值判斷） ==========
def predict_image(model, image_path, threshold=confidence_threshold):
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        max_prob, predicted = torch.max(probs, 1)

        if max_prob.item() < threshold:
            return -1, "Uncertain"  # 新增：低信心分類為 Uncertain

    return predicted.item(), class_names[predicted.item()]

# ========== 對資料夾中圖片進行分類並移動 ==========
def classify_and_move_images(model, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    # 建立子資料夾（含 Uncertain）
    for cls in class_names + ["Uncertain"]:
        os.makedirs(os.path.join(folder_path, cls), exist_ok=True)

    all_preds = []
    all_files = []

    print(f"\n🚀 開始分類資料夾內圖片：{folder_path}")
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)

        # 跳過非圖片檔案或資料夾
        if not os.path.isfile(fpath) or not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        pred_idx, pred_class = predict_image(model, fpath)

        # 若為不確定分類，分到 Uncertain 資料夾
        dest_class = pred_class if pred_class in class_names else "Uncertain"
        dest_path = os.path.join(folder_path, dest_class, fname)

        print(f"📷 {fname} → 分類為 {dest_class}")
        shutil.move(fpath, dest_path)

        all_preds.append(pred_idx)
        all_files.append(fname)

    print("\n✅ 所有圖片已分類完畢。")

# ========== 主程式 ==========
if __name__ == "__main__":
    model = load_model('best_resnet18_model.pth')
    test_folder = './data/test2'
    classify_and_move_images(model, test_folder)
