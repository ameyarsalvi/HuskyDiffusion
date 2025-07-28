import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os 


class CustomDataset(Dataset):
    def __init__(self, csv_file, base_dir, image_transform=None, input_seq=25, output_seq=100):
        super().__init__()
        self.raw_data = pd.read_csv(csv_file)
        self.data = self.raw_data  # Step 1: Subsample every 5th row
        self.image_transform = image_transform
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.base_dir = base_dir

    def __len__(self):
        return len(self.data) - (self.input_seq + self.output_seq)

    def __getitem__(self, idx):
        input_rows = self.data.iloc[idx : idx + self.input_seq]
        output_rows = self.data.iloc[idx + self.input_seq : idx + self.input_seq + self.output_seq]

        # Image & IMU input sequence
        image_paths = input_rows['fix_img_path'].tolist()
        pos_x = input_rows['pose_X'].values
        pos_x = pos_x - pos_x[0]
        pos_y = input_rows['pose_Y'].values
        pos_y = pos_y - pos_y[0]

        # Output sequence (future trajectory)
        cmd_v = output_rows['cmd_v'].values
        cmd_omg = output_rows['cmd_v'].values

        # Load and transform image sequence
        image_sequence = []
        for path in image_paths:
            full_path = os.path.join(self.base_dir, path)  # Join base_dir + relative path
            img = Image.open(full_path).convert('RGB')
            if self.image_transform:
                img = self.image_transform(img)
            image_sequence.append(img)

        image_sequence = torch.stack(image_sequence, dim=0)  # shape: (T, C, H, W)


        actions = torch.tensor(
            np.stack((cmd_v, cmd_omg), axis=-1),
            dtype=torch.float32
        )

        return {
            'images': image_sequence,              # (input_seq, C, H, W)
            'posX': torch.tensor(pos_x),           # (input_seq,)
            'posY': torch.tensor(pos_y),           # (input_seq,)
            'actions': actions                     # (output_seq, 2)
        }
