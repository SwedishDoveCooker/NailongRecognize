from flask import Flask, request, jsonify
from PIL import Image, ImageSequence
import torch
# from io import BytesIO
from img import load_model, predict_frame, test_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './nailong.pth'
model = load_model(model_path, device)

app = Flask(__name__)

def predict_single_image(image_stream):
    try:
        image = Image.open(image_stream)
        if image.format == 'GIF':
            gif = image
            for frame in ImageSequence.Iterator(gif):
                frame = frame.convert('RGB')
                if predict_frame(frame, model, test_transform, device):
                    return "Prediction: Positive"
            return "Prediction: Negative"
        else:
            image = image.convert('RGB')
            result = predict_frame(image, model, test_transform, device)
            return "奶龙" if result else "非奶龙"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    try:
        result = predict_single_image(image_file.stream)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7001)
