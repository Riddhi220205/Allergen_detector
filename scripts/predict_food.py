import torch
from torchvision import models, transforms
from PIL import Image
import argparse
from torchvision.datasets import ImageFolder
import os

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data/food-101"
MODEL_PATH = "models/food101_resnet50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ----------------------------
# LOAD CLASSES
# ----------------------------
train_data = ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
classes = train_data.classes
num_classes = len(classes)
print("Classes used:", classes)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_food(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probs, 3)  # top-3 predictions

    print(f"\nPredictions for {image_path}:")
    for i in range(3):
        print(f"{i+1}. {classes[top_catid[i].item()]} - {top_prob[i].item()*100:.2f}%")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to the image")
    args = parser.parse_args()
    predict_food(args.image)


