import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False

def preprocess(raw_dir, processed_img_dir, processed_annotations_path, fer_au_codes_path, config_path, img_size=None, use_mtcnn=False, limit=None):
    os.makedirs(processed_img_dir, exist_ok=True)
    with open(config_path, "r") as f:
        cfg = json.load(f)
    output_size = int(img_size or cfg.get("img_size", 128))
    detector = MTCNN() if (use_mtcnn and MTCNN_AVAILABLE) else None
    use_mtcnn_flag = detector is not None
    img_list = os.listdir(raw_dir)
    if limit is not None:
        img_list = img_list[:int(limit)]
    processed_img_names = []
    for img_name in tqdm(img_list):
        img_path = os.path.join(raw_dir, img_name)
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if use_mtcnn_flag:
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
            pil_img = pil_img.resize((output_size, output_size), Image.Resampling.BICUBIC)
            save_path = os.path.join(processed_img_dir, img_name)
            pil_img.save(save_path)
            processed_img_names.append(img_name)
        except Exception:
            continue
    if os.path.exists(fer_au_codes_path):
        fer_au_codes = np.load(fer_au_codes_path)
    else:
        fer_au_codes = np.zeros((len(processed_img_names), 16))
        os.makedirs(os.path.dirname(fer_au_codes_path), exist_ok=True)
        np.save(fer_au_codes_path, fer_au_codes)
    annotations = []
    for i, img_name in enumerate(processed_img_names):
        annotations.append({
            "img_path": f"images/{img_name}",
            "au_code": fer_au_codes[i % len(fer_au_codes)].tolist(),
            "role_type": "real",
            "frame_paths": []
        })
    os.makedirs(os.path.dirname(processed_annotations_path), exist_ok=True)
    with open(processed_annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print("预处理完成！")
    print(f"- 处理后图片数量：{len(processed_img_names)}")
    print(f"- 标注文件路径：{processed_annotations_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/celebA")
    parser.add_argument("--processed_img_dir", default="data/processed/images")
    parser.add_argument("--processed_annotations_path", default="data/processed/annotations.json")
    parser.add_argument("--fer_au_codes_path", default="data/processed/fer_au_codes.npy")
    parser.add_argument("--config_path", default="config/train_config.json")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--use_mtcnn", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    preprocess(
        raw_dir=args.raw_dir,
        processed_img_dir=args.processed_img_dir,
        processed_annotations_path=args.processed_annotations_path,
        fer_au_codes_path=args.fer_au_codes_path,
        config_path=args.config_path,
        img_size=args.img_size,
        use_mtcnn=args.use_mtcnn,
        limit=args.limit
    )
