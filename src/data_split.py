import os
import shutil
import random
from tqdm import tqdm

def split_by_class_folders(
    source_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例總和必須是 1.0"
    random.seed(seed)

    class_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in class_folders:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        split_sets = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_files in split_sets.items():
            split_class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for filename in tqdm(split_files, desc=f"{class_name} - {split_name}", leave=False):
                src_path = os.path.join(class_path, filename)
                dst_path = os.path.join(split_class_dir, filename)
                shutil.copy2(src_path, dst_path)

    print("✅ 所有類別資料分割完成！")

# ✅ 使用方式
split_by_class_folders(
    source_dir="./data/dentistry",           # 你的原始資料夾，包含 PA、PANO、other
    output_dir="./data/dentistry_split",    # 分割後的輸出位置
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
