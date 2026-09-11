import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h, c = hidden_state

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device
        )

        c = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device
        )

        return h, c


class StormSenseConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels=1,
        hidden_channels=16,
        output_channels=1
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels
        )

        self.output_conv = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1
        )

    def forward(self, x, future_steps=12):
        """
        x shape:
        (batch, time, channels, height, width)
        """

        batch_size, time_steps, channels, height, width = x.shape

        h, c = self.cell.init_hidden(
            batch_size,
            height,
            width,
            x.device
        )

        # Process observed frames
        for t in range(time_steps):
            h, c = self.cell(x[:, t], (h, c))

        # Generate future frames
        predictions = []

        current_input = x[:, -1]

        for _ in range(future_steps):
            h, c = self.cell(current_input, (h, c))

            prediction = torch.sigmoid(
                self.output_conv(h)
            )

            predictions.append(prediction)

            current_input = prediction

        return torch.stack(predictions, dim=1)
