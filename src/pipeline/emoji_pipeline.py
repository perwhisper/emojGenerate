import json
import torch
import os
import sys
sys.path.append("../../")  # 根目录

from src.model.generator import EmojiGenerator
from src.data_process.au_encoder import AUEncoder
from src.meme.meme_matcher import HotMemeMatcher
from src.meme.static_renderer import StaticTextRenderer
from utils.img_utils import tensor2img
from PIL import ImageFilter
from utils.model_utils import load_model

class EmojiGeneratorPipeline:
    def __init__(self, config_path):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.device = torch.device(self.config["device"])

        # 加载训练好的模型
        self.static_gen = load_model(
            EmojiGenerator(au_dim=16, num_res_blocks=9), 
            self._get_latest_static_gen_path(),
            self.device
        )
        self.static_gen.eval()  # 推理模式

        # 初始化工具类
        self.au_encoder = AUEncoder("./config/emotion_au_map.json")
        self.meme_matcher = HotMemeMatcher()
        self.static_renderer = StaticTextRenderer()

    def _get_latest_static_gen_path(self):
        dir_path = "./models/static_gen"
        os.makedirs(dir_path, exist_ok=True)
        candidates = [f for f in os.listdir(dir_path) if f.startswith("static_gen_epoch_") and f.endswith(".pth")]
        if not candidates:
            raise FileNotFoundError("No checkpoints found in ./models/static_gen")
        def epoch_num(name):
            return int(name.split("_")[-1].split(".")[0])
        latest = max(candidates, key=epoch_num)
        return os.path.join(dir_path, latest)

    def preprocess_img(self, img_path):
        from PIL import Image
        from torchvision import transforms
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((self.config["img_size"], self.config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = transform(img).unsqueeze(0).to(self.device)
        return img

    def generate(self, img_path, emotion_text, intensity=1.0, role_type="real", custom_meme_text=None):
        """生成静态表情包主函数"""
        # 1. 预处理输入
        img = self.preprocess_img(img_path)
        # 2. 情感转AU编码
        au_code = self.au_encoder.text2au(emotion_text, intensity).unsqueeze(0).to(self.device)
        # 3. 角色类型编码
        role_type_code = torch.tensor([self.config["role_map"][role_type]]).to(self.device)

        # 4. 生成表情包底图（推理模式，无梯度）
        with torch.no_grad():
            fake_img = self.static_gen(img, au_code, role_type_code)
            fake_img_pil = tensor2img(fake_img)
            fake_img_pil = fake_img_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            # 保存临时底图
            temp_path = "./temp/temp_static.png"
            os.makedirs("./temp", exist_ok=True)
            fake_img_pil.save(temp_path)

        # 5. 添加热梗配文
        meme_text = custom_meme_text if custom_meme_text else self.meme_matcher.get_hot_meme_text(emotion_text)
        output_path = f"./outputs/static/{emotion_text}_{role_type}.png"
        self.static_renderer.add_text(temp_path, meme_text, emotion_text, output_path)

        # 清理临时文件（可选）
        # os.remove(temp_path)

        return output_path, meme_text

if __name__ == "__main__":
    # 测试生成
    pipeline = EmojiGeneratorPipeline("./config/train_config.json")
    static_path, text = pipeline.generate(
        img_path="./data/processed/test.jpg",
        emotion_text="委屈撇嘴",
        intensity=0.8,
        role_type="real"
    )
    print(f"静态表情包生成完成：{static_path}")
    print(f"配文：{text}")
