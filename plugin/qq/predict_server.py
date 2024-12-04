import gradio as gr
import torch
from PIL import Image, ImageSequence
from img import load_model, predict_frame, test_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './nailong.pth'
model = load_model(model_path, device)

def predict_single_image(image):
    try:
        if isinstance(image, Image.Image) and image.format == 'GIF':
            gif = image
            for frame in ImageSequence.Iterator(gif):
                frame = frame.convert('RGB')
                if predict_frame(frame, model, test_transform, device):
                    return "奶龙"
            return "非奶龙"
        else:
            result = predict_frame(image, model, test_transform, device)
            return "奶龙" if result else "非奶龙"
    except Exception as e:
        return f"Error: {str(e)}"

interface = gr.Interface(fn=predict_single_image, 
                         inputs=gr.Image(type="pil", label="上传图片"),
                         outputs=gr.Textbox(label="预测结果"))

if __name__ == "__main__":
    interface.launch(server_port=7001, share=True)
