import shutil
from pathlib import Path
import pandas as pd

# Load your original CSV
csv_path = r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_real\training_dataset.csv"
df = pd.read_csv(csv_path)

# Define original base path and new image directory
original_base = Path(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_real\images\imgs_bag_1")
target_base = Path(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_real\images")

target_base.mkdir(parents=True, exist_ok=True)

# Move images
for path in df['image_path']:
    filename = Path(path).name
    src = original_base / filename
    dst = target_base / filename
    shutil.move(src, dst)  # use move() if you want to delete original
print("✅ Copied all images to modular 'images/' folder.")
