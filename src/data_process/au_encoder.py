import json
import torch

class AUEncoder:
    def __init__(self, au_map_path):
        with open(au_map_path, "r", encoding="utf-8") as f:
            self.emotion_au_map = json.load(f)
        # 固定16个AU维度（匹配模型输入）
        self.all_aus = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU12", "AU15", "AU17", 
                        "AU20", "AU23", "AU25", "AU26", "AU30", "AU43", "AU63", "AU64"]

    def text2au(self, emotion_text, intensity=1.0):
        """将情感文本转为16维AU张量"""
        if emotion_text not in self.emotion_au_map:
            raise ValueError(f"不支持的情感：{emotion_text}，可选：{list(self.emotion_au_map.keys())}")
        au_dict = self.emotion_au_map[emotion_text]
        # 乘以情感强度
        au_dict = {au: val * intensity for au, val in au_dict.items()}
        # 补全所有AU维度（缺失的填0）
        for au in self.all_aus:
            if au not in au_dict:
                au_dict[au] = 0.0
        # 转为张量
        return torch.tensor([au_dict[au] for au in self.all_aus], dtype=torch.float32)

if __name__ == "__main__":
    # 测试代码
    encoder = AUEncoder("../../config/emotion_au_map.json")
    print(encoder.text2au("委屈撇嘴", 0.8))