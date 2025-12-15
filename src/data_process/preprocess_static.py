import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    from mtcnn import MTCNN
    USE_MTCNN = True
except Exception:
    USE_MTCNN = False

detector = MTCNN() if USE_MTCNN else None

RAW_CELEBA_DIR = "data/raw/celebA"
PROCESSED_IMG_DIR = "data/processed/images"
PROCESSED_ANNOTATIONS_PATH = "data/processed/annotations.json"
FER_AU_CODES_PATH = "data/processed/fer_au_codes.npy"
CONFIG_PATH = "config/train_config.json"

os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)
OUTPUT_SIZE = int(cfg.get("img_size", 128))

print("开始裁剪人脸图片...")
img_list = os.listdir(RAW_CELEBA_DIR)
processed_img_names = []

for img_name in tqdm(img_list):
    img_path = os.path.join(RAW_CELEBA_DIR, img_name)
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if USE_MTCNN:
            results = detector.detect_faces(img)
            if len(results) == 0:
                h, w = img.shape[:2]
                s = min(h, w)
                cy, cx = h // 2, w // 2
                y1 = max(0, cy - s // 2)
                x1 = max(0, cx - s // 2)
                y2 = y1 + s
                x2 = x1 + s
            else:
                x1, y1, width, height = results[0]['box']
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = x1 + max(1, width)
                y2 = y1 + max(1, height)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                size = int(max(width, height) * 1.3)
                x1 = max(0, cx - size // 2)
                y1 = max(0, cy - size // 2)
                x2 = min(img.shape[1], x1 + size)
                y2 = min(img.shape[0], y1 + size)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                h, w = img.shape[:2]
                s = min(h, w)
                cy, cx = h // 2, w // 2
                y1 = max(0, cy - s // 2)
                x1 = max(0, cx - s // 2)
                y2 = y1 + s
                x2 = x1 + s
            else:
                (x, y, w0, h0) = faces[0]
                cx = x + w0 // 2
                cy = y + h0 // 2
                size = int(max(w0, h0) * 1.3)
                x1 = max(0, cx - size // 2)
                y1 = max(0, cy - size // 2)
                x2 = min(img.shape[1], x1 + size)
                y2 = min(img.shape[0], y1 + size)
        face_img = img[y1:y2, x1:x2]
        pil_img = Image.fromarray(face_img)
        pil_img = pil_img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BICUBIC)
        save_path = os.path.join(PROCESSED_IMG_DIR, img_name)
        pil_img.save(save_path)
        processed_img_names.append(img_name)
    except Exception:
        continue

# --------------------------
# 步骤2：生成标注文件
# --------------------------
print("生成标注文件...")
if os.path.exists(FER_AU_CODES_PATH):
    fer_au_codes = np.load(FER_AU_CODES_PATH)
else:
    fer_au_codes = np.zeros((len(processed_img_names), 16))
    np.save(FER_AU_CODES_PATH, fer_au_codes)

# 构建标注列表
annotations = []
for i, img_name in enumerate(processed_img_names):
    annotations.append({
        "img_path": f"images/{img_name}",
        "au_code": fer_au_codes[i % len(fer_au_codes)].tolist(),
        "role_type": "real",
        "frame_paths": []
    })

# 保存标注文件
with open(PROCESSED_ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"预处理完成！")
print(f"- 处理后图片数量：{len(processed_img_names)}")
print(f"- 标注文件路径：{PROCESSED_ANNOTATIONS_PATH}")
