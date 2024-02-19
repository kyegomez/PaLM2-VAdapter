[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Palm2 Adapter
Implementation of "PaLM2-VAdapter:" from the multi-modal model paper: "PaLM2-VAdapter: Progressively Aligned Language Model Makes a Strong Vision-language Adapter".

This model uses a perceiver resampler with a depth of 1 + a tiny palm to efficiently learn the features behind the images and then map them  to the same space as the big model.

## install
`$ pip install palm2-vadapter`


## usage
```
import torch
from palm_vadapter.main import PaLM2VAdapter

# Random text and image tensors
text = torch.randint(0, 1000, (1, 32), dtype=torch.long)


# Image tensor
img = torch.randn(1, 3, 224, 224)

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
```


# License
MIT
