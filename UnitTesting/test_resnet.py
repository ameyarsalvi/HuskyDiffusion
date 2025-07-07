import pytest
import torch
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.resnet import get_resnet  # Import ResNet module

@pytest.mark.parametrize("resnet_variant, expected_dim", [
    ("resnet18", 512),
    ("resnet34", 512),
    ("resnet50", 2048),
    ("resnet101", 2048),
])
def test_resnet_initialization(resnet_variant, expected_dim):
    """Test if ResNet initializes correctly and has expected output dimensions."""
    model = get_resnet(name=resnet_variant)
    model.eval()  # Set model to evaluation mode

    # Create a random image input (Batch size=1, Channels=3, Height=96, Width=96)
    x = torch.randn(1, 3, 96, 96)
    
    # Forward pass through the model
    with torch.no_grad():
        features = model(x)

    # Check output shape: (batch_size, feature_dim)
    assert features.shape == (1, expected_dim), f"Expected output shape (1, {expected_dim}), but got {features.shape}"

def test_resnet_fc_layer_removed():
    """Ensure that ResNet’s classification layer is removed (replaced with Identity)."""
    model = get_resnet("resnet18")

    # Check if the final fc layer is Identity
    assert isinstance(model.fc, torch.nn.Identity), "Expected ResNet.fc to be nn.Identity"
