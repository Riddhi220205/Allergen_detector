import torch
import torch.nn as nn
from torchvision import models

def build_model():
    print("🔧 Loading pretrained ResNet50...")
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Replace last layer for Food-101 classification
    num_classes = 101
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f"✅ Model modified for {num_classes} classes.")

    return model


if __name__ == "__main__":
    model = build_model()
    print(model)  # Show the architecture
    # Optional: save it for later
    torch.save(model.state_dict(), "models/food101_resnet50.pth")
    print("💾 Model saved to models/food101_resnet50.pth")
