import gradio as gr
import os
import sys
sys.path.append("../../")  # 根目录

from src.pipeline.emoji_pipeline import EmojiGeneratorPipeline

# 初始化生成管道
pipeline = EmojiGeneratorPipeline("./config/train_config.json")

def generate_emoji(img, emotion_text, intensity, role_type, custom_meme_text):
    """Gradio调用的生成函数"""
    # 保存用户上传的图片
    img_path = "./temp/user_input.jpg"
    os.makedirs("./temp", exist_ok=True)
    img.save(img_path)
    
    # 生成表情包
    output_path, meme_text = pipeline.generate(
        img_path=img_path,
        emotion_text=emotion_text,
        intensity=intensity/100,  # 滑块0-100转0-1
        role_type=role_type,
        custom_meme_text=custom_meme_text
    )
    return output_path, meme_text

def random_switch_meme_text(emotion_text):
    """随机切换热梗配文"""
    return pipeline.meme_matcher.get_hot_meme_text(emotion_text)

# 构建Web界面
with gr.Blocks(title="静态个性化表情包生成系统") as demo:
    gr.Markdown("# 🎭 静态个性化表情包生成系统")
    gr.Markdown("### 上传图片 → 选择情感 → 生成专属表情包（支持热梗配文）")

    with gr.Row():
        # 左侧：输入区
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="上传人物/角色图片", height=300)
            emotion_text = gr.Dropdown(
                choices=list(pipeline.meme_matcher.corpus.keys()),
                label="选择情感类型",
                value="委屈撇嘴",
                interactive=True
            )
            intensity = gr.Slider(
                minimum=0, maximum=100, label="情感强度（0=自然，100=夸张）",
                value=80, step=5
            )
            role_type = gr.Radio(
                choices=["real", "cartoon", "handdrawn"],
                label="表情包风格",
                value="real",
                interactive=True
            )
            
            gr.Markdown("#### ✨ 热梗配文设置")
            custom_meme_text = gr.Textbox(
                label="自定义配文（留空自动匹配热梗）",
                placeholder="例如：谁懂啊，真的会谢",
                lines=2
            )
            switch_btn = gr.Button("🔄 随机切换热梗配文")
            current_meme_text = gr.Textbox(
                label="当前匹配热梗",
                interactive=False,
                lines=1
            )

            generate_btn = gr.Button("🚀 生成表情包", variant="primary")
        
        # 右侧：输出区
        with gr.Column(scale=1):
            output_img = gr.Image(
                type="filepath", label="生成的静态表情包",
                height=300
            )
            meme_text_display = gr.Textbox(
                label="最终配文",
                interactive=False,
                lines=2
            )

    # 绑定事件
    switch_btn.click(random_switch_meme_text, [emotion_text], [current_meme_text])
    generate_btn.click(
        generate_emoji,
        inputs=[img_input, emotion_text, intensity, role_type, custom_meme_text],
        outputs=[output_img, meme_text_display]
    )

if __name__ == "__main__":
    # 启动Demo（本地访问：http://localhost:7860）
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)