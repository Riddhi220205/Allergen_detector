
import torch
import torch.nn as nn
from torchvision import models

# Load your trained model architecture
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 101)  # same as build_model.py

# Load the trained weights
model.load_state_dict(torch.load("models/food101_resnet50.pth", map_location="cpu"))

model.eval()  # set to evaluation mode
print("✅ Model loaded successfully and ready for inference!")
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example test image
img = Image.open("data/test/pizza/pizza1.jpeg")  # change path to your test image
img = transform(img).unsqueeze(0)

outputs = model(img)
_, predicted = outputs.max(1)
print(f"Predicted class index: {predicted.item()}")
