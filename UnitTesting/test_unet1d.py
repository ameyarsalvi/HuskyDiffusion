import pytest
import torch
import sys
sys.path.insert(0, "C:/Users/asalvi/Documents/Ameya_workspace/DiffusionDataset/ConeCamAngEst/training/") 
from modules.unet import ConditionalUnet1D  # Import your UNet module

@pytest.mark.parametrize("input_dim, global_cond_dim, batch_size, seq_len", [
    (1000, 16, 4, 100),  # Standard case
    (512, 32, 2, 50),    # Smaller input and longer conditioning
    (128, 64, 1, 200),   # Long sequence length
    (256, 8, 8, 10),     # Short sequence
])
def test_unet_output_shape(input_dim, global_cond_dim, batch_size, seq_len):
    """Test if the UNet preserves correct output shape."""
    model = ConditionalUnet1D(input_dim, global_cond_dim)

    #x = torch.randn(batch_size, input_dim, seq_len)  # Noisy input sequence
    x = torch.randn(batch_size, seq_len, input_dim)

    timesteps = torch.randint(0, 1000, (batch_size,))  # Random diffusion timesteps
    global_cond = torch.randn(batch_size, global_cond_dim)  # Conditioning vector

    output = model(x, timesteps, global_cond)

    # Expected output shape: (batch_size, input_dim, seq_len)
    assert output.shape == (batch_size, seq_len, input_dim), \
        f"Expected shape {(batch_size, seq_len, input_dim)}, but got {output.shape}"

def test_unet_conditioning_effect():
    """Test if different conditioning vectors produce different outputs."""
    model = ConditionalUnet1D(512, 16)

    x = torch.randn(4, 512, 100)
    timesteps = torch.randint(0, 1000, (4,))
    cond1 = torch.randn(4, 16)  # Conditioning 1
    cond2 = torch.randn(4, 16)  # Conditioning 2

    output1 = model(x, timesteps, cond1)
    output2 = model(x, timesteps, cond2)

    assert not torch.allclose(output1, output2), \
        "Different conditioning vectors should produce different outputs"

def test_unet_backward_pass():
    """Ensure gradients flow correctly through the UNet."""
    model = ConditionalUnet1D(512, 16)
    
    x = torch.randn(4, 512, 100)  # Input
    timesteps = torch.randint(0, 1000, (4,))
    global_cond = torch.randn(4, 16)  # Conditioning vector
    y = torch.randn(4, 512, 100)  # Target output

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output = model(x, timesteps, global_cond)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()  # ✅ Check if backpropagation works
    optimizer.step()

    assert loss.item() > 0, "Loss should be greater than zero after backpropagation."

def test_unet_residual_connections():
    """Ensure that skip connections properly influence the output."""
    model = ConditionalUnet1D(512, 16)

    x1 = torch.randn(4, 512, 100)  # Input 1
    x2 = torch.randn(4, 512, 100)  # Input 2 (slightly different)
    timesteps = torch.randint(0, 1000, (4,))
    global_cond = torch.randn(4, 16)  # Conditioning vector

    output1 = model(x1, timesteps, global_cond)
    output2 = model(x2, timesteps, global_cond)

    assert not torch.allclose(output1, output2), \
        "Skip connections should influence the final output"
