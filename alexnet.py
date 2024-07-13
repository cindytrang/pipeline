import torch
from torchvision import models, transforms
from PIL import Image
import ast

alexnet = models.alexnet(pretrained=True)
alexnet.eval() 

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_imagenet_classes(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        classes_dict = ast.literal_eval(content)
    return classes_dict

def analyze_image(image_path, imagenet_classes):
    img = Image.open(image_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0) 

    with torch.no_grad():
        out = alexnet(batch_t)

    _, index = torch.max(out, 1)
    index = index.item() 

    label = imagenet_classes[index]
    
    return label, index 

imagenet_classes = load_imagenet_classes('imagenet_classes.txt')

image_path = 'image.jpg'
label, index = analyze_image(image_path, imagenet_classes)
print(f"Predicted label: {label}, index: {index}")