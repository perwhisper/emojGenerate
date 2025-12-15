import cv2
import numpy as np
from PIL import Image

def img2tensor(img_path, size=256):
    """图片转Tensor（适配模型输入）"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return img

def tensor2img(tensor):
    """Tensor转PIL图片（保存/展示用）"""
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img + 1) / 2 * 255.0
    img = img.astype(np.uint8)
    return Image.fromarray(img)