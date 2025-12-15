import json
import pandas as pd
import numpy as np
import os

# 添加路径校验
config_path = "config/emotion_au_map.json"
csv_path = "data/raw/fer2013/fer2013.csv"
save_path = "data/processed/fer2013_au_codes_small.npy"

# 检查输入文件
if not os.path.exists(config_path):
    print(f"❌ 缺少配置文件：{config_path}")
    exit(1)
if not os.path.exists(csv_path):
    print(f"❌ 缺少FER数据集：{csv_path}")
    exit(1)

# 确保输出目录存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 原有逻辑
with open(config_path, "r", encoding="utf-8") as f:
    emotion_au_map = json.load(f)

fer_emotion_map = {0:"生气皱眉",1:"厌恶",2:"恐惧",3:"开心大笑",4:"难过哭泣",5:"惊讶张嘴",6:"佛系淡定"}
all_aus = ["AU1","AU2","AU4","AU6","AU7","AU12","AU15","AU17","AU20","AU23","AU25","AU26","AU30","AU43","AU63","AU64"]

df = pd.read_csv(csv_path).head(500)
au_list = []
for idx, row in df.iterrows():
    emotion = fer_emotion_map[int(row["emotion"])]
    au_dict = emotion_au_map.get(emotion, emotion_au_map["佛系淡定"])
    au_code = [au_dict.get(au, 0.0) for au in all_aus]
    au_list.append(au_code)

np.save(save_path, np.array(au_list))
print(f"✅ 精简版AU编码生成完成（500条），保存路径：{save_path}")
