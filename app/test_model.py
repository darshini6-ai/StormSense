import sys
import torch

sys.path.append(".")

from preprocessing import create_training_sample
from models.convlstm import StormSenseConvLSTM


# Load training sample
past, future, event_id = create_training_sample(index=0)

# Convert to tensors
x = torch.tensor(past, dtype=torch.float32)
y = torch.tensor(future, dtype=torch.float32)

# (T,H,W) -> (B,T,C,H,W)
x = x.unsqueeze(0).unsqueeze(2)
y = y.unsqueeze(0).unsqueeze(2)

# Create model
model = StormSenseConvLSTM(
    input_channels=1,
    hidden_channels=16,
    output_channels=1
)

# Forward pass
with torch.no_grad():
    prediction = model(
        x,
        future_steps=12
    )

print("Event ID:", event_id)
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Prediction shape:", prediction.shape)
print("Prediction min:", prediction.min().item())
print("Prediction max:", prediction.max().item())
