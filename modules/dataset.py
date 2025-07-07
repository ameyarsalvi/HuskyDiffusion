import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, csv_file, image_transform=None, input_seq=25, output_seq=100):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.input_seq = input_seq
        self.output_seq = output_seq

    def __len__(self):
        return len(self.data) - (self.input_seq + self.output_seq)

    def __getitem__(self, idx):
        rows = self.data.iloc[idx:idx + (self.input_seq +self.output_seq)]
        image_paths = rows.iloc[0:self.input_seq]['rs_image_no'].tolist()
        IMU_v = rows.iloc[0:self.input_seq]['odom_VelLinX'].values
        IMU_omg = rows.iloc[0:self.input_seq]['IMU_fil_AngZ'].values
        pos_x = rows.iloc[self.input_seq:]['pos_x'].values
        pos_y = rows.iloc[self.input_seq:]['pos_y'].values

        image_sequence = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            if self.image_transform:
                img = self.image_transform(img)
            image_sequence.append(img)

        image_sequence = torch.stack(image_sequence, dim=0)  # shape: (T, C, H, W)

        # Normalize pose values by subtracting the first pose
        pos_x_normalized = pos_x - pos_x[0]
        pos_y_normalized = pos_y - pos_y[0]

        # Sequence of normalized positions as training labels
        actions = torch.tensor(
            np.stack((pos_x_normalized, pos_y_normalized), axis=-1),
            dtype=torch.float32
        )

        return {
        'images': image_sequence,              # (T, C, H, W)
        'imu_v': torch.tensor(IMU_v),          # (T,)
        'imu_omg': torch.tensor(IMU_omg),      # (T,)
        'actions': actions                     # (output_seq,T)
    }
