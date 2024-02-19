import torch
from palm_vadapter.main import VisualAdapter

# Create random tensors
x = torch.randn(1, 32, 512)
img = torch.randn(1, 3, 224, 224)

# Initialize VisualAdapter model
model = VisualAdapter(
    dim=512,
    depth=6,
    heads=8,
    dropout=0.1,
    dim_head=64,
    ff_mult=4,
    image_size=224,
    patch_size=16,
)

# Forward pass through the model
out = model(x, img)

# Print the shape of the output
print(out.shape)
