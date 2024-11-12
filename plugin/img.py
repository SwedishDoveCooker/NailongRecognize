import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2

def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    
    for image_file in image_files:
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
            img = Image.open(image_file).convert('RGB')
            jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
            img.save(jpg_path, 'JPEG')
            print(f"Converted {image_file} to {jpg_path}")
            os.remove(image_file)
            print(f"Deleted original file: {image_file}")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_model(model_path, device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_frame(frame, model, transform, device):
    model.eval()
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frame)
        _, pred = torch.max(output, 1)
    return pred.item() == 1

def predict_image_or_gif(file_path, model, transform, device):
    model.eval()
    if file_path.lower().endswith('.gif'):
        gif = Image.open(file_path)
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert('RGB')
            if predict_frame(frame, model, transform, device):
                return True
        return False
    else:
        image = Image.open(file_path).convert('RGB')
        return predict_frame(image, model, transform, device)

def predict_video(video_path, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    found = False
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if predict_frame(pil_image, model, transform, device):
            found = True
            print(f"Video: {video_path}, Frame {frame_count}: True")
        frame_count += 1
    cap.release()
    if not found:
        print(f"Video: {video_path}, Prediction: False")
    return found

def run_predictions(input_dir, model, transform, device):
    final_results = []
    convert_images_to_jpg(input_dir)
    all_files = glob.glob(os.path.join(input_dir, '*.*'))
    for file_path in all_files:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            predict_video(file_path, model, transform, device)
        else:
            result = predict_image_or_gif(file_path, model, transform, device)
            final_results.append(result)
            # print(f"File: {file_path}, Prediction: {'True' if result else 'False'}")
    return True if any(final_results) else False


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_path = './nailong.pth'
    input_dir = './input'
    model = load_model(model_path, device)
    run_predictions(input_dir, model, test_transform, device)

if __name__ == '__main__':
    main()