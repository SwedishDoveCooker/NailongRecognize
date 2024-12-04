import torch
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50().to(device)
model.fc = torch.nn.Linear(model.fc.in_features, 2).to(device) 
model.load_state_dict(torch.load("./nailong.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "./test_attack.png"
image = Image.open(image_path).convert("RGB")
image.save("example.jpg") 
image = Image.open("example.jpg").convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)

output = model(input_image)
original_label = output.argmax(dim=1).item()
print(f"原图像标签: {original_label}")

epsilon = 0.1
alpha = 0.01  
steps = 10

adv_image = input_image.clone().detach().requires_grad_(True)
for _ in range(steps):
    output = model(adv_image)
    loss = F.cross_entropy(output, torch.tensor([original_label]).to(device))
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        adv_image = adv_image + alpha * adv_image.grad.sign()
        adv_image = torch.min(torch.max(adv_image, input_image - epsilon), input_image + epsilon)
        adv_image = torch.clamp(adv_image, 0, 1)
        adv_image.requires_grad = True

adv_output = model(adv_image)
adv_label = adv_output.argmax(dim=1).item()
print(f"对抗样本标签: {adv_label}")

def save_image(tensor, filename):
    tensor = tensor.squeeze().detach().cpu()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    tensor = torch.clamp(tensor, 0, 1)
    img = Image.fromarray((tensor.numpy() * 255).astype("uint8"))
    img.save(filename)

save_image(input_image, "original_image.jpg")
save_image(adv_image, "adversarial_image.jpg")
print("原图像和对抗样本已保存为'original_image.jpg'和'adversarial_image.jpg'。")

def imshow(tensor, title):
    tensor = tensor.squeeze().detach().cpu()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    tensor = torch.clamp(tensor, 0, 1)
    plt.imshow(tensor)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
imshow(input_image, title=f"原图像 (Label: {original_label})")
plt.subplot(1, 2, 2)
imshow(adv_image, title=f"对抗样本 (Label: {adv_label})")
plt.show()
