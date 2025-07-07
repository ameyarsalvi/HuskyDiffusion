import pytest
import torch
import os
import pandas as pd
from torchvision import transforms

import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.dataset import CustomDataset  # Import your dataset module

# Mock CSV file path
CSV_PATH = "mock_dataset.csv"

# Create a mock dataset
@pytest.fixture(scope="module", autouse=True)
def create_mock_csv():
    data = {
        "rs_image_no": ["mock_image.jpg"] * 10,  # Mock image file
        "pos_x": list(range(10)),
        "pos_y": list(range(10, 20))
    }
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    yield
    os.remove(CSV_PATH)  # Cleanup after test

# Mock image for testing
@pytest.fixture(scope="module", autouse=True)
def create_mock_image():
    from PIL import Image
    img = Image.new('RGB', (96, 96))  # Create a blank image
    img.save("mock_image.jpg")
    yield
    os.remove("mock_image.jpg")  # Cleanup after test

def test_dataset_loading():
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    
    dataset = CustomDataset(CSV_PATH, image_transform=transform, sequence_length=5)
    
    assert len(dataset) == 6  # 10 - sequence_length + 1
    sample = dataset[0]
    
    assert isinstance(sample, dict), "Dataset should return a dictionary"
    assert 'image' in sample and 'actions' in sample, "Keys missing in sample"

    assert isinstance(sample["image"], torch.Tensor), "Image should be a PyTorch Tensor"
    assert sample["image"].shape == (3, 96, 96), "Incorrect image shape, expected (3, 96, 96)"

    assert isinstance(sample["actions"], torch.Tensor), "Actions should be a PyTorch Tensor"
    assert sample["actions"].ndim == 1, "Actions tensor should be 1D"

def test_dataset_normalization():
    dataset = CustomDataset(CSV_PATH, sequence_length=5)

    sample = dataset[0]
    pos_x = sample["actions"].numpy()[::2]  # Extract x positions
    pos_y = sample["actions"].numpy()[1::2]  # Extract y positions
    
    assert pos_x[0] == 0, "First x position should be 0 after normalization"
    assert pos_y[0] == 0, "First y position should be 0 after normalization"
