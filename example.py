import torch
from palm_vadapter.main import PaLM2VAdapter

# Random text and image tensors
text = torch.randint(0, 1000, (1, 32), dtype=torch.long)

# Image tensor [batch, channels, height, width] -- # Create a random tensor with shape (batch_size, channels, height, width)
img = torch.randn(1, 3, 224, 224)

# Video tensor -- # Create a random tensor with shape (batch_size, channels, frames, height, width)
# video = torch.randn(1, 3, 16, 224, 224)

# Initialize PaLM2VAdapter model
model = PaLM2VAdapter(
    tiny_dim=512,
    dim=512,
    num_tokens=10000,
    seq_length=32,
    depth=6,
    heads=8,
    image_size=224,
    patch_size=16,
)

# Forward pass through the model
out = model(text, img)

# Print the shape of the output
print(out.shape)
