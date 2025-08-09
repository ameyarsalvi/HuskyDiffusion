import pandas as pd
from pathlib import Path
import os

dir = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig"
# Load your original CSV
csv_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig\merged_output_cone_rig.csv"
df = pd.read_csv(csv_path)

# Define the base directory where images are stored (relative to dataset root)
image_dir_name = "images"

# Convert absolute paths to relative ones
df['image'] = df['image'].apply(
    lambda p: Path(p).name  # just get the filename, e.g., 'imgs_bag_1_105.jpg'
).apply(
    lambda fname: f"{image_dir_name}/{fname}"  # prepend with relative folder
)
'''
# Convert absolute paths to relative ones
df['mv_img_path'] = df['mv_img_path'].apply(
    lambda p: Path(p).name  # just get the filename, e.g., 'imgs_bag_1_105.jpg'
).apply(
    lambda fname: f"{image_dir_name}/{fname}"  # prepend with relative folder
)
'''

# Save the new CSV (modular and cross-platform)
df.to_csv(os.path.join(dir,"mod_output_cone_rig.csv"), index=False)
print("✅ Saved modular path CSV as 'modular_data.csv'")
