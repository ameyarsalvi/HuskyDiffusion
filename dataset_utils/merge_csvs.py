import pandas as pd
import os
from pathlib import Path

# Define the folder where all your CSV files are located
csv_folder = Path(r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\DriveRigHuskyLaneKeep")

# Get list of all CSV files in the folder
csv_files = list(csv_folder.glob("*.csv"))

# Load and concatenate all CSVs
merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

save_path =r"C:\Users\asalvi\Documents\Ameya_workspace\DiffusionDataset\training_dataset\cone_path_drive_rig"
# Save the merged DataFrame to a new file
merged_df.to_csv(os.path.join(save_path, "merged_output_cone_rig.csv"), index=False)

print(f"✅ Merged {len(csv_files)} files into 'merged_output.csv'")
