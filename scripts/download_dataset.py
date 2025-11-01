from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.Food101(root='./data', split='train', transform=transform, download=True)
test_data = datasets.Food101(root='./data', split='test', transform=transform, download=True)
