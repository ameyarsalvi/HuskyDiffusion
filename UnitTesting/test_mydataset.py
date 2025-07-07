import torch
import pandas as pd
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,"C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.dataset import CustomDataset  # Import your dataset module

# Path to your actual CSV file
CSV_PATH = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\csv_files\TSyn_data_filtered.csv"

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalization
])

# Load Dataset
dataset = CustomDataset(csv_file=CSV_PATH, image_transform=transform, sequence_length=100)

# Select 5 random indices
random_indices = random.sample(range(len(dataset)), 5)

# Plot samples
for idx in random_indices:
    sample = dataset[idx]

    image = sample["image"].permute(1, 2, 0).numpy()  # Convert to HxWxC format
    image = (image * 0.5) + 0.5  # Reverse normalization for visualization

    #conditioning = sample["conditioning"].item()
    actions = sample["actions"].numpy()

    # Reshape actions back to (pos_x, pos_y) format
    pos_x = actions[::2]  # Every even index
    pos_y = actions[1::2]  # Every odd index

    # Plot Image
    plt.figure(figsize=(10, 4))
    
    # Image Plot
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image")
    plt.axis("off")

    # Plot Positions
    plt.subplot(1, 2, 2)
    plt.plot(pos_x, pos_y, marker="o", linestyle="-", color="blue")
    plt.xlabel("Normalized Pos X")
    plt.ylabel("Normalized Pos Y")
    plt.title("Action Sequence (Trajectory)")
    plt.grid()

    plt.show()
