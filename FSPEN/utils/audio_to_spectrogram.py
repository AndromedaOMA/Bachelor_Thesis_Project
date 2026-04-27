import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

# Define your file paths
path_base = '../../samples/4sec_samples/with_multy_head_attention/'
file_map = {
    "Clean": path_base + 'clean/clean_2_22.wav',
    "Noisy": path_base + 'noisy/noisy_2_22.wav',
    "Enhanced": path_base + 'enhanced/enhanced_2_22.wav'
}

# Create a figure with 3 stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
titles = ["Clean", "Noisy", "Enhanced"]

for i, title in enumerate(titles):
    # Load each file individually
    sample_rate, samples = wavfile.read(file_map[title])

    # Calculate spectrogram with higher resolution
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=1024)

    # Convert to dB scale for visibility
    spectrogram_db = 10 * np.log10(spectrogram + 1e-10)

    # Plot using pcolormesh
    # vmin and vmax ensure the color scale is identical across all three for fair comparison
    im = axes[i].pcolormesh(times, frequencies, spectrogram_db,
                            shading='gouraud', cmap='magma')

    axes[i].set_title(f"{title} Signal")
    axes[i].set_ylabel('Frequency [Hz]')

# Label the shared X-axis
axes[2].set_xlabel('Time [sec]')

# Add a colorbar to the side
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Intensity [dB]')

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
