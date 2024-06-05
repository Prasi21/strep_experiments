import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define constants
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 2  # Healthy and not healthy
NUM_EPOCHS = 25
BATCH_SIZE = 20
LEARNING_RATE = 0.0001

# Define data transformations
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load images and labels using ImageFolder
train_dataset = ImageFolder(root="./datasets/Augmented/Trainfolder/", transform=data_transforms)
val_dataset = ImageFolder(root="./datasets/Augmented/Valfolder/", transform=data_transforms)
test_dataset = ImageFolder(root="./datasets/Original/Testfolder/", transform=data_transforms)

# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



timestp = time.strftime("%Y%m%d-%H%M%S")

# New classifier must already be added
def finetune_pretrained(base_model, out_dir=f"model-{timestp}.pt"):
    # Freeze base model layers
    for param in base_model.features.parameters():
        param.requires_grad = False

    # Move model to device
    base_model = base_model.to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.0001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        base_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")


    # Optionally, unfreeze some layers of the convolutional base for fine-tuning
    for param in base_model.features.parameters():
        param.requires_grad = True

    # Redefine optimizer for fine-tuning
    optimizer = optim.Adam(base_model.parameters(), lr=0.0001)


    # Fine-tuning loop
    for epoch in range(NUM_EPOCHS):
        base_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}")




    # Evaluate the model
    base_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = base_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(base_model.state_dict(), out_dir)

# Load the pretrained model
base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
num_ftrs = base_model.classifier[0].in_features

base_model.classifier = nn.Sequential(
    nn.Linear(in_features=num_ftrs, out_features=4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=1),
    nn.Sigmoid()
)
print(base_model)

# Modify the model
finetune_pretrained(base_model)
print("HIIII")

# print(models.resnet50(weights=models.ResNet50_Weights.DEFAULT))
# # Add new classification layers
# num_ftrs = base_model.fc.in_features
# base_model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 1),
#     nn.Sigmoid()
# )
