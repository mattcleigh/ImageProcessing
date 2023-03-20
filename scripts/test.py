import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(12, 12),
)
model = model.to("cuda")

torch.set_float32_matmul_precision("medium")

model = torch.compile(model, mode="default")

model(torch.randn((12, 12), device="cuda"))
print("yay")
