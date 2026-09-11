import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def init_hidden(
        self,
        batch_size,
        height,
        width,
        device
    ):

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


class StormSenseConvLSTMv3(nn.Module):

    def __init__(
        self,
        input_channels=1,
        hidden_channels=32,
        output_channels=1
    ):
        super().__init__()

        # --------------------------------
        # Spatial encoder
        # --------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                input_channels,
                16,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )

        # --------------------------------
        # Temporal ConvLSTM
        # --------------------------------

        self.convlstm = ConvLSTMCell(
            input_channels=32,
            hidden_channels=hidden_channels
        )

        # --------------------------------
        # Spatial decoder
        # --------------------------------

        self.decoder = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                output_channels,
                kernel_size=1
            )
        )

    def decode(self, h, height, width):

        output = self.decoder(h)

        output = F.interpolate(
            output,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        output = torch.sigmoid(output)

        return output

    def forward(
        self,
        x,
        future_frames=None,
        future_steps=12,
        teacher_forcing_ratio=0.0
    ):
        """
        x:
            (batch, time, channels, height, width)

        future_frames:
            Optional ground-truth future sequence.
            Shape:
            (batch, future_time, channels, height, width)

        future_steps:
            Number of future frames to generate.

        teacher_forcing_ratio:
            Probability of using the actual future frame
            as the next input during training.

        Returns:
            (batch, future_steps, channels, height, width)
        """

        batch_size, time_steps, channels, height, width = x.shape

        # --------------------------------
        # Initialize hidden state
        # --------------------------------

        first_encoded = self.encoder(x[:, 0])

        _, _, encoded_height, encoded_width = first_encoded.shape

        h, c = self.convlstm.init_hidden(
            batch_size,
            encoded_height,
            encoded_width,
            x.device
        )

        # --------------------------------
        # Process observed frames
        # --------------------------------

        for t in range(time_steps):

            encoded = self.encoder(x[:, t])

            h, c = self.convlstm(
                encoded,
                (h, c)
            )

        # --------------------------------
        # Generate future frames
        # --------------------------------

        predictions = []

        current_input = x[:, -1]

        for step in range(future_steps):

            # Decode current hidden state
            prediction = self.decode(
                h,
                height,
                width
            )

            predictions.append(prediction)

            # --------------------------------
            # Choose next input
            # --------------------------------

            use_teacher_forcing = (
                self.training
                and future_frames is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )

            if use_teacher_forcing:

                current_input = future_frames[:, step]

            else:

                current_input = prediction

            # Encode next input
            encoded = self.encoder(current_input)

            # Update temporal state
            h, c = self.convlstm(
                encoded,
                (h, c)
            )

        return torch.stack(
            predictions,
            dim=1
        )
