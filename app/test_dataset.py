from app.sevir_dataset import SEVIRVILDataset


FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"


dataset = SEVIRVILDataset(
    file_path=FILE,
    input_frames=12,
    target_frames=12,
    image_size=128,
    max_samples=30
)


print("Total sliding-window samples:", len(dataset))

print("\nFirst 5 windows:")

for i in range(5):

    sequence_index, start_frame = dataset.samples[i]

    print(
        f"Sample {i + 1}: "
        f"Sequence={sequence_index}, "
        f"Start frame={start_frame}"
    )


past, future = dataset[0]

print("\nFirst sample:")
print("Past shape:", past.shape)
print("Future shape:", future.shape)
print("Past range:", past.min().item(), "to", past.max().item())
print("Future range:", future.min().item(), "to", future.max().item())


print("\nSample 26:")

sequence_index, start_frame = dataset.samples[25]

print(
    f"Sequence={sequence_index}, "
    f"Start frame={start_frame}"
)
