import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the image
image_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\ConeCamAngEst\imgs_bag_9\imgs_bag_9_24.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initial RGB bounds for brown color
initial_lower = [50, 30, 10]  # Adjust these values to target brown shades
initial_upper = [200, 180, 120]

# Create the figure and axis
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plt.subplots_adjust(bottom=0.35)

# Create sliders for RGB bounds
ax_r_lower = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_g_lower = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_b_lower = plt.axes([0.2, 0.15, 0.65, 0.03])

ax_r_upper = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_g_upper = plt.axes([0.2, 0.06, 0.65, 0.03])
ax_b_upper = plt.axes([0.2, 0.02, 0.65, 0.03])

slider_r_lower = Slider(ax_r_lower, 'R Lower', 0, 255, valinit=initial_lower[0], valstep=1)
slider_g_lower = Slider(ax_g_lower, 'G Lower', 0, 255, valinit=initial_lower[1], valstep=1)
slider_b_lower = Slider(ax_b_lower, 'B Lower', 0, 255, valinit=initial_lower[2], valstep=1)

slider_r_upper = Slider(ax_r_upper, 'R Upper', 0, 255, valinit=initial_upper[0], valstep=1)
slider_g_upper = Slider(ax_g_upper, 'G Upper', 0, 255, valinit=initial_upper[1], valstep=1)
slider_b_upper = Slider(ax_b_upper, 'B Upper', 0, 255, valinit=initial_upper[2], valstep=1)

# Function to update the mask and segmented image
def update(val):
    lower_bound = np.array([slider_r_lower.val, slider_g_lower.val, slider_b_lower.val], dtype=np.uint8)
    upper_bound = np.array([slider_r_upper.val, slider_g_upper.val, slider_b_upper.val], dtype=np.uint8)

    # Create a mask for brown shades
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    ax[0].cla()
    ax[0].set_title("Mask")
    ax[0].imshow(mask, cmap='gray')
    ax[0].axis('off')

    ax[1].cla()
    ax[1].set_title("Segmented Image")
    ax[1].imshow(segmented)
    ax[1].axis('off')

    fig.canvas.draw_idle()

# Attach the update function to sliders
slider_r_lower.on_changed(update)
slider_g_lower.on_changed(update)
slider_b_lower.on_changed(update)

slider_r_upper.on_changed(update)
slider_g_upper.on_changed(update)
slider_b_upper.on_changed(update)

# Initial display
update(None)
plt.show()
