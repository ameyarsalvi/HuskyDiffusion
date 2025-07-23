import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from segmentation_models_pytorch import Unet
import torch


# Load the U-Net model with a ResNet34 encoder
model = Unet(
    encoder_name="resnet34",       # Encoder architecture
    encoder_weights="imagenet",   # Pre-trained on ImageNet
    classes=21,                    # Number of output classes
    activation=None                # No activation to keep raw logits
)
model.eval()

# Load the image
image_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\imgs_bag_9\imgs_bag_9_24.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Convert to RGB for compatibility with PyTorch
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image_rgb).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
segmentation = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Map segmentation classes to colors
def decode_segmentation(segmentation):
    label_colors = np.array([
        [0, 0, 0],         # Class 0: Background
        [128, 0, 0],       # Class 1
        [0, 128, 0],       # Class 2
        [128, 128, 0],     # Class 3
        [0, 0, 128],       # Class 4
        [128, 0, 128],     # Class 5
        [0, 128, 128],     # Class 6
        [128, 128, 128],   # Class 7
        [64, 0, 0],        # Class 8
        [192, 0, 0],       # Class 9
        [64, 128, 0],      # Class 10
        [192, 128, 0],     # Class 11
        [64, 0, 128],      # Class 12
        [192, 0, 128],     # Class 13
        [64, 128, 128],    # Class 14
        [192, 128, 128],   # Class 15
        [0, 64, 0],        # Class 16
        [128, 64, 0],      # Class 17
        [0, 192, 0],       # Class 18
        [128, 192, 0],     # Class 19
        [0, 64, 128],      # Class 20
    ])
    return label_colors[segmentation]

segmented_image = decode_segmentation(segmentation)

# Display the original and segmented images
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()
