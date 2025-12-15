import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet50, vgg19, VGG19_Weights, ResNet50_Weights
from tqdm import tqdm
import sys
# 兼容从项目根或任何工作目录启动：将项目根加入 sys.path
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from src.data_process.dataset import EmojiDataset
from src.model.generator import EmojiGenerator
from src.model.discriminator import EmojiDiscriminator
from utils.model_utils import save_model, set_requires_grad

class EmojiTrainer:
    def __init__(self, config_path):
        # 加载配置
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.device = torch.device(self.config["device"])
        # RTX 4090 优化与精度设置
        if self.device.type == "cuda":
            torch.cuda.set_device(self.config.get("cuda_device", 0))
            torch.backends.cudnn.benchmark = True
            if self.config.get("tf32", True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            self.precision = self.config.get("precision", "bf16")
            self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            self.scaler = GradScaler(enabled=(self.precision == "fp16"))
        else:
            self.precision = "fp32"
            self.amp_dtype = None
            self.scaler = GradScaler(enabled=False)

        # 初始化模型
        self.static_gen = EmojiGenerator(au_dim=16).to(self.device)
        self.discriminator = EmojiDiscriminator(au_dim=16).to(self.device)
        if self.config.get("use_torch_compile", False):
            self.static_gen = torch.compile(self.static_gen)
            self.discriminator = torch.compile(self.discriminator)

30→        # ArcFace身份提取器（预训练，冻结参数）
31→        self.arcface = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.arcface.fc = nn.Linear(2048, 512)
        self.arcface = self.arcface.to(self.device)
        set_requires_grad(self.arcface, False)
        self.arcface.eval()

        # 损失函数
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.au_loss = nn.MSELoss()             # AU编码损失
        self.identity_loss = nn.L1Loss()        # 身份损失
        self.feature_loss = nn.MSELoss()        # 特征匹配损失
        self.lambda_perc = self.config.get("lambda_perc", 1.0)
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device)
        set_requires_grad(self.vgg, False)
        self.vgg.eval()
        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 优化器
        self.optimizer_G = optim.Adam(
            self.static_gen.parameters(),
            lr=self.config["lr_g"], betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["lr_d"], betas=(0.5, 0.999)
        )

        # 加载数据集
        self.dataset = EmojiDataset(
            data_root="./data/processed",
            config=self.config,
            role_type="all",
            is_train=True
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=int(self.config.get("num_workers", max(1, (os.cpu_count() or 4) // 2))),
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=int(self.config.get("num_workers", max(1, (os.cpu_count() or 4) // 2))) > 0
        )

    def train_static_generator(self):
        """训练静态生成器（核心）"""
        self.static_gen.train()
        self.discriminator.train()

        for epoch in range(self.config["epochs"]):
            total_d_loss = 0.0
            total_g_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                # 加载批次数据
                img, au_code, _, role_type = batch
                mem_format = torch.channels_last if self.config.get("channels_last", True) else torch.contiguous_format
                img = img.to(self.device, non_blocking=True).to(memory_format=mem_format)
                au_code = au_code.to(self.device, non_blocking=True)
                role_type = role_type.to(self.device, non_blocking=True)

                # ---------------------
                # 1. 训练判别器
                # ---------------------
                self.optimizer_D.zero_grad()
                with autocast(dtype=self.amp_dtype, enabled=(self.device.type == "cuda" and self.precision in ["bf16", "fp16"])):  # 混合精度
                    # 真实样本
                    real_pred, real_au, real_id_feat = self.discriminator(img)
                    real_label = torch.ones_like(real_pred, device=self.device)
                    fake_label = torch.zeros_like(real_pred, device=self.device)

                    d_real_loss = self.adversarial_loss(real_pred, real_label)
                    d_real_au_loss = self.au_loss(real_au, au_code)

                    # 伪造样本（生成器输出，detach避免梯度传递）
                    fake_img = self.static_gen(img, au_code, role_type)
                    fake_pred, fake_au, fake_id_feat = self.discriminator(fake_img.detach())
                    d_fake_loss = self.adversarial_loss(fake_pred, fake_label)

                    # 判别器总损失
                    d_loss = d_real_loss + d_fake_loss + self.config["lambda_au"] * d_real_au_loss

                # 反向传播+更新
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.optimizer_D)
                self.scaler.update()

                # ---------------------
                # 2. 训练生成器
                # ---------------------
                self.optimizer_G.zero_grad()
                with autocast(dtype=self.amp_dtype, enabled=(self.device.type == "cuda" and self.precision in ["bf16", "fp16"])):
                    # 伪造样本（重新计算，保留梯度）
                    fake_pred, fake_au, fake_id_feat = self.discriminator(fake_img)
                    # 对抗损失（欺骗判别器）
                    g_adv_loss = self.adversarial_loss(fake_pred, real_label)
                    # AU编码损失（匹配目标情感）
                    g_au_loss = self.au_loss(fake_au, au_code)

                    # 身份损失（保留人脸特征）
                    with torch.no_grad():
                        real_id_feat_arc = self.arcface(img).detach()
                    fake_id_feat_arc = self.arcface(fake_img)
                    g_id_loss = self.identity_loss(fake_id_feat_arc, real_id_feat_arc)

                    # 特征匹配损失（提升生成质量）
                    g_feat_loss = self.feature_loss(fake_id_feat, real_id_feat.detach())
                    img_01 = img * 0.5 + 0.5
                    fake_01 = fake_img * 0.5 + 0.5
                    img_v = (img_01 - self.imnet_mean) / self.imnet_std
                    fake_v = (fake_01 - self.imnet_mean) / self.imnet_std
                    real_vgg = self.vgg(img_v)
                    fake_vgg = self.vgg(fake_v)
                    g_perc_loss = self.identity_loss(fake_vgg, real_vgg)

                    # 生成器总损失
                    g_loss = (g_adv_loss + self.config["lambda_au"] * g_au_loss + 
                              self.config["lambda_id"] * g_id_loss + self.config["lambda_feat"] * g_feat_loss +
                              self.lambda_perc * g_perc_loss)

                # 反向传播+更新
                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.optimizer_G)
                self.scaler.update()

                # 累计损失（日志用）
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                pbar.set_postfix({"D Loss": d_loss.item(), "G Loss": g_loss.item()})

            # 每轮结束打印日志
            avg_d_loss = total_d_loss / len(self.dataloader)
            avg_g_loss = total_g_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} | Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")
            
            # 按间隔保存模型
            if (epoch + 1) % self.config["save_interval"] == 0:
                save_model(self.static_gen, "./models/static_gen/static_gen", epoch+1)
                save_model(self.discriminator, "./models/discriminator/discriminator", epoch+1)

if __name__ == "__main__":
    trainer = EmojiTrainer("./config/train_config.json")
    trainer.train_static_generator()
