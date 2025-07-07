import pytest
import torch
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.conv_residual import ConditionalResidualBlock1D  # Import the module

@pytest.mark.parametrize("in_channels, out_channels, cond_dim, kernel_size, batch_size, seq_len", [
    (16, 32, 8, 3, 4, 100),  # Standard case
    (32, 64, 16, 5, 2, 50),  # Larger feature maps
    (64, 128, 32, 7, 1, 200), # Long sequence length
    (8, 16, 4, 1, 8, 10),    # Edge case: kernel_size=1
    (1, 1, 2, 3, 5, 20)      # Edge case: Single-channel input/output
])
def test_residual_block_output_shape(in_channels, out_channels, cond_dim, kernel_size, batch_size, seq_len):
    """Test if the Residual block maintains correct output shape."""
    model = ConditionalResidualBlock1D(in_channels, out_channels, cond_dim, kernel_size)

    x = torch.randn(batch_size, in_channels, seq_len)  # Input signal
    cond = torch.randn(batch_size, cond_dim)  # Conditioning vector
    output = model(x, cond)

    assert output.shape == (batch_size, out_channels, seq_len), \
        f"Expected shape {(batch_size, out_channels, seq_len)}, but got {output.shape}"

def test_residual_block_conditioning():
    """Test if the conditioning vector correctly modifies the output."""
    model = ConditionalResidualBlock1D(16, 32, 8, 3)
    
    x = torch.randn(4, 16, 100)  # Input
    cond1 = torch.randn(4, 8)  # Conditioning vector 1
    cond2 = torch.randn(4, 8)  # Conditioning vector 2

    output1 = model(x, cond1)
    output2 = model(x, cond2)

    assert not torch.allclose(output1, output2), "Different conditioning vectors should produce different outputs"

def test_residual_block_backward_pass():
    """Test if gradients flow correctly through the residual block."""
    model = ConditionalResidualBlock1D(16, 32, 8, 3)
    
    x = torch.randn(4, 16, 100)  # Input signal
    cond = torch.randn(4, 8)  # Conditioning vector
    y = torch.randn(4, 32, 100)  # Target output

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output = model(x, cond)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()  # ✅ Ensure backpropagation works
    optimizer.step()

    assert loss.item() > 0, "Loss should be greater than zero after backpropagation."

def test_residual_block_conditioning_mismatch():
    """Ensure an error is raised when conditioning size is incorrect."""
    model = ConditionalResidualBlock1D(16, 32, 8, 3)

    x = torch.randn(4, 16, 100)  # Input
    cond_wrong_dim = torch.randn(4, 4)  # Incorrect conditioning dimension

    with pytest.raises(RuntimeError):
        model(x, cond_wrong_dim)  # Should raise an error due to dimension mismatch
