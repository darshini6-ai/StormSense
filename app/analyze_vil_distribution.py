import numpy as np

from app.sevir_dataset import SEVIRVILDataset


dataset = SEVIRVILDataset(
    "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
)

print("VIL distribution analysis")
print("=" * 60)

sample_count = min(200, len(dataset))

max_values = []
mean_values = []
high_pixel_ratios = []

for i in range(sample_count):

    past, future = dataset[i]

    future_np = future.numpy()[:, 0]

    for frame in future_np:

        max_values.append(float(frame.max()))
        mean_values.append(float(frame.mean()))

        high_ratio = np.mean(frame >= 0.10)
        high_pixel_ratios.append(float(high_ratio))


max_values = np.array(max_values)
mean_values = np.array(mean_values)
high_pixel_ratios = np.array(high_pixel_ratios)


print(f"Samples analysed: {sample_count}")
print(f"Future frames analysed: {len(max_values)}")

print()
print("Per-frame maximum VIL")
print(f"  Mean   : {max_values.mean():.4f}")
print(f"  Median : {np.median(max_values):.4f}")
print(f"  P25    : {np.percentile(max_values, 25):.4f}")
print(f"  P75    : {np.percentile(max_values, 75):.4f}")
print(f"  P90    : {np.percentile(max_values, 90):.4f}")
print(f"  P95    : {np.percentile(max_values, 95):.4f}")
print(f"  Maximum: {max_values.max():.4f}")

print()
print("Per-frame mean VIL")
print(f"  Mean   : {mean_values.mean():.4f}")
print(f"  Median : {np.median(mean_values):.4f}")
print(f"  P90    : {np.percentile(mean_values, 90):.4f}")

print()
print("Fraction of pixels with VIL >= 0.10")
print(f"  Mean   : {high_pixel_ratios.mean():.4f}")
print(f"  Median : {np.median(high_pixel_ratios):.4f}")
print(f"  P75    : {np.percentile(high_pixel_ratios, 75):.4f}")
print(f"  P90    : {np.percentile(high_pixel_ratios, 90):.4f}")
print(f"  Maximum: {high_pixel_ratios.max():.4f}")

print()
print("=" * 60)
print("Corrected VIL distribution analysis complete.")
