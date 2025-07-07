import pytest
import torch
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.pose_embedding import SinusoidalPosEmb  # Import Pose Embedding module


@pytest.mark.parametrize("dim", [4, 8, 16, 32])  # Test different embedding sizes
def test_pose_embedding_output_shape(dim):
    """Test if the embedding has the correct output shape."""
    model = SinusoidalPosEmb(dim)

    x = torch.tensor([[1.0], [5.0], [10.0]])  # Input timesteps
    output = model(x).squeeze(1)  # ✅ Remove extra singleton dimension

    assert output.shape == (3, dim), f"Expected output shape (3, {dim}), but got {output.shape}"

def test_pose_embedding_value_range():
    """Test if the embeddings stay within the expected range (-1 to 1)."""
    model = SinusoidalPosEmb(16)

    x = torch.tensor([[0.0], [1.0], [10.0], [100.0]])  # Different timesteps
    output = model(x).squeeze(1)  # ✅ Remove extra dimension

    assert torch.all(output >= -1) and torch.all(output <= 1), "Embeddings should be in range [-1, 1]"

def test_pose_embedding_variation():
    """Test if different inputs generate different embeddings."""
    model = SinusoidalPosEmb(16)

    x1 = torch.tensor([[1.0]])
    x2 = torch.tensor([[2.0]])

    output1 = model(x1).squeeze(1)  # ✅ Remove extra dimension
    output2 = model(x2).squeeze(1)

    assert not torch.allclose(output1, output2), "Embeddings for different inputs should not be identical"

def test_pose_embedding_frequency_scaling():
    """Test if higher dimensions oscillate slower than lower dimensions."""
    model = SinusoidalPosEmb(16)

    x = torch.tensor([[50.0]])  # Single timestep
    output = model(x).squeeze(0).squeeze(0).numpy()  # ✅ Ensure correct shape

    # The first dimensions should oscillate faster than later ones
    assert abs(output[0] - output[1]) > abs(output[14] - output[15]), \
        "Lower dimensions should oscillate more rapidly than higher dimensions"
