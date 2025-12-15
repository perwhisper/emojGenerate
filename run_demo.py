from src.demo.gradio_demo import demo

if __name__ == "__main__":
    # 启动Web Demo
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)