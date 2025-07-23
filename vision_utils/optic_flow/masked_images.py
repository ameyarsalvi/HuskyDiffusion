import cv2
import numpy as np
import os

# Set the RGB bounds for brown shades
lower_bound = np.array([97, 85, 12], dtype=np.uint8)
upper_bound = np.array([255, 180, 115], dtype=np.uint8)

# Input and output directories
input_folder = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\imgs_bag_9"  # Replace with the path to your folder containing images
output_folder = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\masked_imgs_bag_9"  # Replace with the path to save masked images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error reading {filename}, skipping.")
            continue

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a mask for the brown shades
        mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Blend the original and masked images (50% intensity each)
        blended_image = cv2.addWeighted(image, 0.1, masked_image, 0.9, 0)

        # Save the blended image
        cv2.imwrite(output_path, blended_image)
        print(f"Processed and saved: {output_path}")

print("Processing complete. All masked images have been saved.")
