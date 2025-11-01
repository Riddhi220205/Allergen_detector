import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os

def main():
    # ----------------------------
    # CONFIG
    # ----------------------------
    DATA_DIR = "data/food-101"
    MODEL_DIR = "models"
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_SUBSET = True
    SUBSET_SIZE = 2000

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------
    # TRANSFORMS
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # ----------------------------
    # DATA
    # ----------------------------
    train_data = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)

    if USE_SUBSET:
        train_data = Subset(train_data, list(range(min(SUBSET_SIZE, len(train_data)))))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # <--- set 0 on Windows
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

    classes = train_data.dataset.classes if USE_SUBSET else train_data.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes. Training on {len(train_loader.dataset)} images.")

    # ----------------------------
    # MODEL
    # ----------------------------
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    # ----------------------------
    # LOSS & OPTIMIZER
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "food101_resnet50.pth"))
    print("✅ Training complete and model saved.")

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()

