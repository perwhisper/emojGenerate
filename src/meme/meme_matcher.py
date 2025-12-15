import json
import random

class HotMemeMatcher:
    def __init__(self, corpus_path="./assets/hot_meme_corpus.json"):
        self.corpus = json.load(open(corpus_path, encoding="utf-8"))
        # 情感别名映射（简化用户输入）
        self.emotion_alias = {
            "开心": "开心大笑", "生气": "生气皱眉", "委屈": "委屈撇嘴",
            "惊讶": "惊讶张嘴", "难过": "难过哭泣", "害羞": "害羞脸红"
        }
        self.emotion_alias["希望"] = "开心大笑"

    def get_hot_meme_text(self, emotion_text, random_seed=None):
        """根据情感返回随机热梗配文"""
        # 别名转换
        if emotion_text in self.emotion_alias:
            emotion_text = self.emotion_alias[emotion_text]
        # 无匹配则返回默认文案
        if emotion_text not in self.corpus:
            return f"主打一个{emotion_text}😝"
        # 固定随机种子（可选）
        if random_seed:
            random.seed(random_seed)
        return random.choice(self.corpus[emotion_text])

    def update_corpus(self, emotion_text, new_meme_text):
        """更新热梗语料库"""
        if emotion_text not in self.corpus:
            self.corpus[emotion_text] = []
        self.corpus[emotion_text].append(new_meme_text)
        json.dump(self.corpus, open("./assets/hot_meme_corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        
