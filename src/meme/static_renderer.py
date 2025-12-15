from PIL import Image, ImageDraw, ImageFont
import os

class StaticTextRenderer:
    def __init__(self, font_path="./assets/simhei.ttf"):
        self.font_path = font_path
        # 不同情感的文字位置映射
        self.pos_map = {
            "开心大笑": "bottom", "生气皱眉": "top_left", "委屈撇嘴": "bottom",
            "惊讶张嘴": "top_right", "难过哭泣": "bottom", "害羞脸红": "top_right",
            "挑眉疑惑": "top_right", "白眼不屑": "top_left", "抿嘴紧张": "bottom",
            "挑眉邪笑": "top_right", "打哈欠困": "bottom", "狂喜蹦跳": "top_right",
            "无奈摊手": "top_right", "摆烂躺平": "bottom", "抓狂崩溃": "top_left",
            "佛系淡定": "bottom", "尴尬抠地": "bottom", "傲娇扭头": "top_left",
            "感动落泪": "bottom", "吃瓜看戏": "top_right", "社恐躲避": "bottom",
            "吹牛得意": "top_right", "emo心碎": "bottom", "摸鱼摆烂": "bottom"
        }
        # 情感别名
        self.emotion_alias = {"开心": "开心大笑", "生气": "生气皱眉", "委屈": "委屈撇嘴"}

    def add_text(self, img_path, text, emotion_text, output_path="./outputs/static/output.png"):
        """给图片添加带描边的文字"""
        # 打开图片
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        # 设置字体（优先用指定字体，失败则用默认）
        font_size = int(img_h * 0.12)  # 文字大小为图片高度的12%
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except:
            font = ImageFont.load_default(size=font_size)

        # 计算文字位置
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 情感别名转换
        if emotion_text in self.emotion_alias:
            emotion_text = self.emotion_alias[emotion_text]
        pos_type = self.pos_map.get(emotion_text, "top_right")

        if pos_type == "top_right":
            pos = (img_w - text_w - 10, 10)  # 右上角（留10px边距）
        elif pos_type == "bottom":
            pos = ((img_w - text_w) // 2, img_h - text_h - 10)  # 底部居中
        elif pos_type == "top_left":
            pos = (10, 10)  # 左上角

        # 绘制文字（先描黑边，再填白字，增强可读性）
        offset = 2  # 描边宽度
        for dx, dy in [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]:
            draw.text((pos[0]+dx, pos[1]+dy), text, font=font, fill=(0, 0, 0))
        draw.text(pos, text, font=font, fill=(255, 255, 255))

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return output_path