import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

FILE = "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"

# Load one sequence
with h5py.File(FILE, "r") as f:
    vil = f["vil"][0].astype(np.float32)
    event_id = f["id"][0].decode()

print("Event ID:", event_id)
print("Sequence shape:", vil.shape)
print("Number of frames:", vil.shape[-1])

fig, ax = plt.subplots(figsize=(7, 7))

image = ax.imshow(vil[:, :, 0], cmap="turbo", vmin=0, vmax=255)
ax.set_title(f"StormSense | Event {event_id} | Frame 0")
ax.set_xlabel("X")
ax.set_ylabel("Y")

def update(frame):
    image.set_data(vil[:, :, frame])
    ax.set_title(
        f"StormSense | Event {event_id} | "
        f"Frame {frame} | Time +{frame * 5} min"
    )
    return [image]

animation = FuncAnimation(
    fig,
    update,
    frames=vil.shape[-1],
    interval=150,
    blit=True
)

plt.show()
