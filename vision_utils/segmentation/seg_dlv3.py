import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from PIL import Image

# Load the DeeplabV3+ model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Load the input image
image_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\imgs_bag_9\imgs_bag_9_24.jpg"
image = Image.open(image_path).convert("RGB")

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0).byte().cpu().numpy()

def decode_segmentation(segmentation):
    label_colors = np.array([
        [0, 0, 0],         # Background
        [0, 255, 0],       # Vegetation
        [139, 69, 19]      # Trees
    ])
    return label_colors[segmentation]


# Decode the segmentation
segmented_image = decode_segmentation(output_predictions)

# Display the original and segmented images
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()
