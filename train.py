import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import ConcatDataset, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
from deduplicator import ProducerConsumer
from preprocessor import convert_images_to_jpg

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

# 此处对图像的操作不可逆, 请提前备份且训练时若未修改数据集仅需开启一次即可
if __name__ == '__main__':
    convert_images_to_jpg('./train_positive')
    convert_images_to_jpg('./train_negative')
    convert_images_to_jpg('./test')
    convert_images_to_jpg('./negative_test')
    convert_images_to_jpg('./input')
    pc = ProducerConsumer()
    pc.run("train_positive/*.jpg")
    pc.run("train_negative/*.jpg")
    pc.run("test/*.jpg")
    pc.run("negative_test/*.jpg")
    pc.run("input/*.jpg")

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
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to('cuda' if torch.cuda.is_available() else 'cpu'))

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
    train_model(model, train_loader, test_loader, epochs=10, optimizer=optimizer, criterion=criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './nailong.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")