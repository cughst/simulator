import torch

checkpoint = torch.load('dqn.pt', map_location='cpu')
print(checkpoint.keys())

