import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


os.makedirs("./datasets/Augmented/Valfolder/Class0", exist_ok=True)
os.makedirs("./datasets/Augmented/Valfolder/Class1", exist_ok=True)
os.makedirs("./datasets/Augmented/Trainfolder/Class0", exist_ok=True)
os.makedirs("./datasets/Augmented/Trainfolder/Class1", exist_ok=True)


# Directory containing the original images
original_dir = "./datasets/Original/Trainfolder"

# Directory to save augmented images
save_dir_root = "./datasets/Augmented/Trainfolder"
os.makedirs(save_dir_root, exist_ok=True)

# Image augmentation parameters

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # Horizontal flip with probability 1.0
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Random translation of up to 5% in both width and height
    transforms.RandomRotation(degrees=10),  # Random rotation within the range of -10 to +10 degrees
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Random zoom within the range of 80% to 120%
    transforms.ColorJitter(brightness=0.1),  # Random brightness adjustment within the range of -0.1 to +0.1
    transforms.ToTensor()
])

# Function to apply augmentation and save images
def augment_images(load_dir, save_dir):
    for filename in os.listdir(load_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):  # Assuming images are in jpg or png format
            img_path = os.path.join(load_dir, filename)
            img = Image.open(img_path)  # Load image
            augmented_images = []
            num_augmented_images = 10  # Number of augmented images per original image
            for i in range(num_augmented_images):
                augmented_img = data_transform(img)
                augmented_images.append(augmented_img)

            # Save augmented images
            base_filename = os.path.splitext(filename)[0]
            for i, augmented_img in enumerate(augmented_images):
                save_path = os.path.join(save_dir, f"{base_filename}_aug_{i}.jpg")
                torchvision.utils.save_image(augmented_img, save_path)

print("Augmenting Training Data")
# Recursively traverse through subdirectories and apply augmentation
for root, dirs, files in os.walk(original_dir):
    for class_dir in dirs:
        class_path = os.path.join(root, class_dir)
        save_class_dir = os.path.join(save_dir_root, class_dir)
        os.makedirs(save_class_dir, exist_ok=True)
        augment_images(class_path, save_class_dir)

print("Augmentation complete.")

# Repeat the same process for validation data
original_dir = "./datasets/Original/Valfolder"
save_dir_root = "./datasets/Augmented/Valfolder"
os.makedirs(save_dir_root, exist_ok=True)

print("Augmenting Validation Data")
for root, dirs, files in os.walk(original_dir):
    for class_dir in dirs:
        class_path = os.path.join(root, class_dir)
        save_class_dir = os.path.join(save_dir_root, class_dir)
        os.makedirs(save_class_dir, exist_ok=True)
        augment_images(class_path, save_class_dir)

print("Augmentation complete.")
