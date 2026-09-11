import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SEVIRVILDataset(Dataset):

    def __init__(
        self,
        file_path,
        input_frames=12,
        target_frames=12,
        image_size=128,
        max_samples=None
    ):

        self.file_path = file_path
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.image_size = image_size

        # Open file temporarily to find number of sequences
        with h5py.File(self.file_path, "r") as f:
            self.num_sequences = f["vil"].shape[0]

        # Each sequence contains 49 frames.
        # Create sliding windows:
        #
        # 12 input + 12 future = 24 frames
        # 49 - 24 + 1 = 26 windows per sequence
        #
        self.windows_per_sequence = (
            49 - input_frames - target_frames + 1
        )

        # Create (sequence_index, starting_frame) pairs
        self.samples = []

        for sequence_index in range(self.num_sequences):

            for start_frame in range(
                self.windows_per_sequence
            ):

                self.samples.append(
                    (sequence_index, start_frame)
                )

        # Limit samples if requested
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sequence_index, start_frame = self.samples[index]

        with h5py.File(self.file_path, "r") as f:

            vil = f["vil"][
                sequence_index,
                :,
                :,
                start_frame:
                start_frame
                + self.input_frames
                + self.target_frames
            ]

        # Original:
        # (H, W, T)
        #
        # Convert to:
        # (T, H, W)

        vil = torch.tensor(
            vil,
            dtype=torch.float32
        )

        vil = vil.permute(2, 0, 1)

        # Normalize VIL
        vil = vil / 255.0

        # Add channel dimension
        # (T, H, W)
        # →
        # (T, 1, H, W)

        vil = vil.unsqueeze(1)

        # Resize spatial dimensions
        # 384x384 → 128x128

        vil = F.interpolate(
            vil,
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        )

        # Split into past and future

        past = vil[
            :self.input_frames
        ]

        future = vil[
            self.input_frames:
            self.input_frames + self.target_frames
        ]

        return past, future
