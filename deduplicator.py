import cv2
import torchvision.models as models
from tqdm import tqdm
import threading
import queue
import shutil
import uuid
import torch
import os
import glob
import concurrent.futures
from torchvision.models.resnet import ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

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