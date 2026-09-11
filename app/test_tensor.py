import torch
from preprocessing import create_training_sample

past, future, event_id = create_training_sample(index=0)

# Convert NumPy arrays to PyTorch tensors
past_tensor = torch.tensor(past, dtype=torch.float32)
future_tensor = torch.tensor(future, dtype=torch.float32)

# Add channel dimension
past_tensor = past_tensor.unsqueeze(1)
future_tensor = future_tensor.unsqueeze(1)

# Add batch dimension
past_tensor = past_tensor.unsqueeze(0)
future_tensor = future_tensor.unsqueeze(0)

print("Event ID:", event_id)
print("Past tensor shape:", past_tensor.shape)
print("Future tensor shape:", future_tensor.shape)
print("Past dtype:", past_tensor.dtype)
print("Future dtype:", future_tensor.dtype)
print("Past min/max:", past_tensor.min().item(), past_tensor.max().item())
print("Future min/max:", future_tensor.min().item(), future_tensor.max().item())
