import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.dataset import CustomDataset  # Import dataset module
from modules.resnet import get_resnet  # Import ResNet encoder


# 📌 Load Dataset
CSV_PATH = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\csv_files\TSyn_data_filtered.csv"

transform = transforms.Compose([
    transforms.Resize((96, 96)),  
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = CustomDataset(CSV_PATH, image_transform=transform, sequence_length=500)

# 📌 Select 5 Random Samples
random_indices = random.sample(range(len(dataset)), 5)
samples = [dataset[idx] for idx in random_indices]

# 📌 Load ResNet Encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = get_resnet("resnet50").to(device)
resnet.eval()

# 📌 Process Images Through ResNet
images = torch.stack([sample["image"] for sample in samples]).to(device)  # Stack into a batch
with torch.no_grad():
    feature_vectors = resnet(images)  # Shape: (5, 512)

# 📌 Convert Feature Vectors to CPU for Visualization
features_np = feature_vectors.cpu().numpy()  # Shape: (5, 512)

# 📌 Plot Results
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, (sample, feat) in enumerate(zip(samples, features_np)):
    # 📌 Reverse Normalize Image for Display
    image = sample["image"].cpu().permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    image = (image * 0.5) + 0.5  # Reverse normalization

    # 📌 Plot Original Image
    axes[0, i].imshow(image)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"Sample {random_indices[i]}")

    # 📌 Plot ResNet Feature Heatmap
    axes[1, i].imshow(feat.reshape(32, 64), cmap="viridis", aspect="auto")  # Reshape for visualization
    axes[1, i].set_title("ResNet Feature Map")
    axes[1, i].axis("off")

plt.suptitle("ResNet Feature Maps (Global Conditioning)")
plt.show()
