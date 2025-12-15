import torch
from src.pipeline.emoji_pipeline import EmojiGeneratorPipeline

# 强制使用CPU（本地无GPU时）
torch.cuda.is_available = lambda: False
device = torch.device("cpu")

if __name__ == "__main__":
    # 初始化生成管道
    pipeline = EmojiGeneratorPipeline("./config/train_config.json")
    
    # 生成示例（替换为自己的图片路径）
    output_path, meme_text = pipeline.generate(
        img_path="./data/processed/test.jpg",  # 本地图片路径
        emotion_text="开心大笑",               # 情感类型
        intensity=0.9,                        # 情感强度
        role_type="cartoon",                  # 风格：real/cartoon/handdrawn
        custom_meme_text=None                 # 自定义配文（None则用热梗）
    )
    
    print("="*50)
    print(f"✅ 表情包生成完成！")
    print(f"📁 保存路径：{output_path}")
    print(f"📝 配文：{meme_text}")
    print("="*50)
