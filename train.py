import os
import glob
from PIL import Image, ImageSequence
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
import cv2
# from .a import *
import torchvision.models as models
from tqdm import tqdm
import threading
import queue
import shutil
import uuid
import concurrent.futures
from torchvision.models.resnet import ResNet18_Weights

def is_duplicate_old(features, feature, thres=0.99):
    if len(features) == 0:
        return False
    for feat in features:
        similarity = F.cosine_similarity(feat, feature, dim=0).item()
        if similarity > thres:
            return True
    return False

def is_duplicate(features, feature, thres=0.99):
    if len(features) < 1000:
        return is_duplicate_old(features, feature, thres=0.99)

    num_vectors = len(features)
    batch_size = 1000
    max_similarity = -1
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(0, num_vectors, batch_size):
            batch_vector = torch.stack(features[i:i+batch_size])
            similarity = F.cosine_similarity(feature.unsqueeze(0), batch_vector, dim=1)
            batch_max_similarity, max_batch_index = torch.max(similarity, dim=0)
            batch_max_similarity = batch_max_similarity.item()
            if batch_max_similarity > thres:
                return True
    return False

class ProducerConsumer:
    def __init__(self):
        self.queue = queue.Queue(maxsize=2000)
        self.extract = FeatureExtract()
        self.features = []

    def produce(self, Path):
        images = glob.glob(Path)
        for img_path in tqdm(images):
            # print(img_path)
            feature = self.extract.feature_extract(img_path)
            self.queue.put([img_path, feature])
        self.queue.put(None)

    def consume(self):
        while True:
            img_info = self.queue.get()
            if img_info is None:
                break
            img_path, feature = img_info
            if is_duplicate(self.features, feature):  
                continue
            else:
                self.features.append(feature)
                base_dir = os.path.dirname(img_path)
                filename = os.path.basename(img_path)
                file_ext = os.path.splitext(filename)[1]
                new_filename = "%s%s" % (uuid.uuid4(), file_ext)
                img_save_path = os.path.join(base_dir, new_filename)
                try:
                    shutil.copy(img_path, img_save_path)
                    # print(f"Image saved to {img_save_path}")
                    os.remove(img_path)
                    # print(f"Original image {img_path} removed")
                except Exception as e:
                    print(f"Failed to save or remove image {img_path}: {e}")

    def run(self, Path):
        producer_thread = threading.Thread(target=self.produce(Path))
        consumer_thread = threading.Thread(target=self.consume)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

class FeatureExtract(object):
    def __init__(self):
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def feature_extract(self, image_path):
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            features = self.resnet(input_batch)
        return features.squeeze()


def process_gif(gif_path, output_dir):
    try:
        gif = Image.open(gif_path)
        first_frame = next(ImageSequence.Iterator(gif))
        first_frame = first_frame.convert('RGB')
        frame_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(gif_path))[0]}.jpg")
        first_frame.save(frame_path, 'JPEG')
        print(f"Saved first frame to {frame_path}")
        
        gif.close()
        os.remove(gif_path)
        print(f"Deleted original GIF file: {gif_path}")
    except PermissionError as e:
        print(f"Failed to delete {gif_path}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {gif_path}: {e}")

def convert_images_to_jpg(directory):
    image_files = glob.glob(os.path.join(directory, '*.*'))
    
    for image_file in image_files:
        if image_file.lower().endswith('.gif'):
            if directory == './input':
                continue
            else:
                process_gif(image_file, directory)
        elif not image_file.lower().endswith(('.jpg', '.jpeg', '.gif', '.mp4', '.avi', '.mov')):
            img = Image.open(image_file).convert('RGB')
            jpg_path = os.path.join(directory, f"{os.path.splitext(os.path.basename(image_file))[0]}.jpg")
            img.save(jpg_path, 'JPEG')
            print(f"Converted {image_file} to {jpg_path}")
            os.remove(image_file)
            print(f"Deleted original file: {image_file}")

class NailongDataset(Dataset):
    def __init__(self, positive_root, negative_root, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for filename in glob.glob(os.path.join(positive_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(1)

        for filename in glob.glob(os.path.join(negative_root, '*.jpg')):
            self.image_paths.append(filename)
            self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

data_augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    data_augmentation_transform
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

'''convert_images_to_jpg('./train_positive')
convert_images_to_jpg('./train_negative')
convert_images_to_jpg('./test')
convert_images_to_jpg('./negative_test')
convert_images_to_jpg('./input')
pc = ProducerConsumer()
pc.run("train_positive/*.jpg")
pc.run("train_negative/*.jpg")
pc.run("test/*.jpg")
pc.run("negative_test/*.jpg")
pc.run("input/*.jpg")'''

if __name__ == '__main__':
    train_dataset = NailongDataset(positive_root='./train_positive', negative_root='./train_negative', transform=train_transform)
    test_dataset = NailongDataset(positive_root='./test', negative_root='./negative_test', transform=test_transform)

def balance_dataset(dataset):
    X = [i for i in range(len(dataset))]
    y = [dataset[i][1] for i in range(len(dataset))]
    unique, counts = np.unique(y, return_counts=True)
    # print(f"Label distribution before balancing: {dict(zip(unique, counts))}")
    X = np.array(X).reshape(-1, 1);y = np.array(y)
    min_class = unique[np.argmin(counts)];min_count = counts[np.argmin(counts)]
    smote = SMOTE(sampling_strategy={label: min_count * 10 if label != min_class else count 
                                    for label, count in zip(unique, counts)}, random_state=42)
    rus = RandomUnderSampler(sampling_strategy={min_class: min_count}, random_state=42)
    pipeline = Pipeline(steps=[('o', smote), ('u', rus)])
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    unique, counts = np.unique(y_resampled, return_counts=True)
    # print(f"Label distribution after balancing: {dict(zip(unique, counts))}")
    balanced_dataset = torch.utils.data.Subset(dataset, X_resampled.flatten())
    return balanced_dataset

if __name__ == '__main__':
    balanced_train_dataset = balance_dataset(train_dataset)

    train_loader = DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to('cuda'))

def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    model.train()
    best_val_loss = float('inf')
    patience = 7
    no_improvement_count = 0

    for stage in range(2):
        if stage == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-5)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            val_loss = evaluate_model(model, val_loader, device)
            print(f'Stage [{stage+1}/2], Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f'Early stopping at Stage [{stage+1}/2], Epoch [{epoch+1}/{epochs}]')
                    break

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    return running_loss / len(test_loader)

if __name__ == '__main__':
    train_model(model, train_loader, test_loader, epochs=10, optimizer=optimizer, criterion=criterion, device='cuda')

    evaluate_model(model, test_loader, device='cuda')

    model_path = './nailong.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")