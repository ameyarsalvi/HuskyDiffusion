import pytest
import torch
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.conv1d import Conv1dBlock  # Import Conv1D module

@pytest.mark.parametrize("inp_channels, out_channels, kernel_size, batch_size, seq_len", [
    (16, 32, 3, 4, 100),  # Standard case
    (32, 64, 5, 2, 50),   # Larger input/output channels
    (64, 128, 7, 1, 200), # Long sequence length
    (8, 16, 1, 8, 10),    # Edge case: kernel_size=1
    (1, 1, 3, 5, 20)      # Edge case: Single-channel input/output
])
def test_conv1d_output_shape(inp_channels, out_channels, kernel_size, batch_size, seq_len):
    """Test if the Conv1D block produces the correct output shape."""
    model = Conv1dBlock(inp_channels, out_channels, kernel_size)

    # Create a dummy input tensor with shape (batch_size, inp_channels, seq_len)
    x = torch.randn(batch_size, inp_channels, seq_len)
    output = model(x)

    # Expected output shape: (batch_size, out_channels, seq_len)
    assert output.shape == (batch_size, out_channels, seq_len), \
        f"Expected shape {(batch_size, out_channels, seq_len)}, but got {output.shape}"

def test_conv1d_forward_pass():
    """Test if the Conv1D block runs without errors."""
    model = Conv1dBlock(16, 32, 3)
    x = torch.randn(4, 16, 100)  # Batch of 4
    output = model(x)
    assert output is not None, "Forward pass failed!"

def test_conv1d_backward_pass():
    """Test if gradients propagate correctly during backpropagation."""
    model = Conv1dBlock(16, 32, 3)
    x = torch.randn(4, 16, 100)  # Input
    y = torch.randn(4, 32, 100)  # Target output

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output = model(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()  # ✅ Ensure backpropagation works
    optimizer.step()

    assert loss.item() > 0, "Loss should be greater than zero after backpropagation."

def test_conv1d_kernel_padding():
    """Test if the kernel size and padding are handled correctly."""
    model = Conv1dBlock(16, 32, 3)  # Kernel size = 3
    x = torch.randn(1, 16, 10)  # Small sequence
    output = model(x)

    # The output sequence length should match input because of padding
    assert output.shape[2] == 10, "Padding should preserve sequence length!"
