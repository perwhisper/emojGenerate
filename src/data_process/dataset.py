import torch
# 强制设置多进程共享策略，解决子进程导入问题
torch.multiprocessing.set_sharing_strategy('file_system')

import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .augment import get_augmentation_pipeline

class EmojiDataset(Dataset):
    def __init__(self, data_root, config, role_type="all", is_train=True):
        self.data_root = data_root
        self.config = config
        self.img_size = config.get("img_size", 64)  # 默认尺寸64，兼容配置缺失
        self.is_train = is_train
        
        # 训练集用数据增强，测试集仅做尺寸调整
        if is_train:
            self.transform = get_augmentation_pipeline(role_type, self.img_size)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # 加载标注文件并做异常处理
        ann_path = os.path.join(data_root, "annotations.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"标注文件不存在: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        
        # 筛选指定角色类型的数据
        if role_type != "all":
            self.annotations = [ann for ann in self.annotations if ann.get("role_type") == role_type]
        if len(self.annotations) == 0:
            raise ValueError(f"未找到 role_type={role_type} 的数据")
        
        # 角色类型映射（兼容配置文件）
        self.role_map = config.get("role_map", {"real": 0, "fake": 1})

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            ann = self.annotations[idx]
            # 拼接图片绝对路径并加载
            img_path = os.path.join(self.data_root, ann["img_path"])
            img = Image.open(img_path).convert('RGB')
            
            # 应用数据变换
            if self.transform:
                img = self.transform(img)
            
            # 转换 AU 编码为 tensor
            au_code = torch.tensor(ann["au_code"], dtype=torch.float32)
            
            # 转换角色类型为 tensor
            role_label = self.role_map.get(ann["role_type"], 0)
            role_type = torch.tensor(role_label, dtype=torch.long)
            
            # 返回值兼容原有训练逻辑
            return img, au_code, [], role_type
        
        except Exception as e:
            # 单条数据出错时返回随机张量，避免训练中断
            print(f"数据加载失败（idx={idx}）: {str(e)}")
            img = torch.randn(3, self.img_size, self.img_size)
            au_code = torch.zeros(16, dtype=torch.float32)
            role_type = torch.tensor(0, dtype=torch.long)
            return img, au_code, [], role_type

# 测试代码：直接运行该文件可验证数据集是否正常
if __name__ == "__main__":
    test_config = {
        "img_size": 64,
        "role_map": {"real": 0}
    }
    dataset = EmojiDataset(
        data_root="data/processed",
        config=test_config,
        role_type="real",
        is_train=False
    )
    print(f"数据集长度: {len(dataset)}")
    img, au, _, role = dataset[0]
    print(f"图片形状: {img.shape}, AU编码形状: {au.shape}, 角色标签: {role.item()}")
