import torch
import os

def save_model(model, save_path, epoch):
    """保存模型到指定路径"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), f"{save_path}_epoch_{epoch}.pth")

def load_model(model, load_path, device):
    """加载训练好的模型"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    return model

def set_requires_grad(model, requires_grad=False):
    """冻结/解冻模型参数"""
    for param in model.parameters():
        param.requires_grad = requires_grad