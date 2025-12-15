from src.training.trainer import EmojiTrainer

if __name__ == "__main__":
    # 初始化训练器并启动训练
    trainer = EmojiTrainer("./config/train_config.json")
    trainer.train_static_generator()