import os
import cv2
import numpy as np
import time

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

def get_image_files(directory, image_type_list):
    image_files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_type_list):
            image_files.append(os.path.join(directory, filename))
    return image_files

def read_image(path):
    image = cv2.imread(path)
    if image is None:
        try:
            image = cv2.imdecode(np.fromfile(file=path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Failed to read image: {path} ({e})")
            return None
    return image

def dhash(image, hash_size=8):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        binary_hash = ''.join(['1' if v else '0' for v in diff.flatten()])
        return int(binary_hash, 2)
    except Exception as e:
        print(f"dHash error: {e}")
        return None

def hamming_distance(hash1, hash2):
    return bin(hash1 ^ hash2).count('1')

def remove_extension_name(file_path):
    return os.path.splitext(file_path)[0]

def get_file_creation_date(file_path):
    try:
        if os.name == 'nt':
            return os.path.getctime(file_path)
        else:
            stat = os.stat(file_path)
            return getattr(stat, 'st_birthtime', stat.st_mtime)
    except:
        return float('inf')

def auto_delete_duplicates(directory, image_type_list=IMAGE_EXTENSIONS, hamming_threshold=0):
    start_time = time.time()
    image_files = get_image_files(directory, image_type_list)
    total_files = len(image_files)
    print(f"Processing {total_files} images...")

    image_hashes = {}
    for i, file_path in enumerate(image_files):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"Processed {i}/{total_files} images ({i/total_files*100:.1f}%) - {elapsed:.1f}s elapsed")
        image = read_image(file_path)
        if image is not None:
            hash_value = dhash(image)
            if hash_value is not None:
                image_hashes[file_path] = hash_value

    # 分群
    grouped_files = []
    used = set()
    files = list(image_hashes.keys())
    for i, f1 in enumerate(files):
        if f1 in used:
            continue
        group = [f1]
        h1 = image_hashes[f1]
        for j in range(i+1, len(files)):
            f2 = files[j]
            if f2 in used:
                continue
            h2 = image_hashes[f2]
            if hamming_distance(h1, h2) <= hamming_threshold:
                group.append(f2)
                used.add(f2)
        used.add(f1)
        if len(group) > 1:
            grouped_files.append(group)

    if not grouped_files:
        print("沒有找到任何重複圖片分組。")
        return

    deleted_count = 0
    for group_idx, group in enumerate(grouped_files, 1):
        sorted_group = sorted(group, key=lambda x: get_file_creation_date(x))
        keeper = sorted_group[0]
        print(f"\n=== 第 {group_idx} 組 (共 {len(grouped_files)} 組) ===")
        print(f"[保留] {keeper} (創建時間：{time.ctime(get_file_creation_date(keeper))})")
        for f in sorted_group[1:]:
            print(f"  - [刪除] {f} (創建時間：{time.ctime(get_file_creation_date(f))})")
            try:
                if os.path.exists(f):
                    os.remove(f)
                    deleted_count += 1
                    print(f"已刪除: {f}")
                base = remove_extension_name(f)
                for ext in ['.json', '.npy']:
                    sidecar = base + ext
                    if os.path.exists(sidecar):
                        os.remove(sidecar)
                        print(f"已刪除關聯檔案: {sidecar}")
            except Exception as e:
                print(f"刪除失敗 {f}: {e}")

    print(f"\n所有分組處理完畢！已自動刪除 {deleted_count} 個重複檔案")
    print(f"總耗時: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    IMAGE_DIR = "data/dentistry_split/val/PANO"  # 修改為你的圖片資料夾
    auto_delete_duplicates(IMAGE_DIR, hamming_threshold=0)
