import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_local_fft_energy(gray_img, window_size=64, step_size=8):
    """
    Slide a window over `gray_img`, take a 2D FFT of each patch and sum
    the magnitude in a mid-to-high radial band. High values mean strong
    periodic structure (streets, building grids).
    """
    H, W = gray_img.shape
    h_steps = (H - window_size) // step_size + 1
    w_steps = (W - window_size) // step_size + 1

    feature_map = np.zeros((h_steps, w_steps))

    # 2D Hanning window: fades each patch to zero at the edges so the FFT
    # does not see a fake step at the boundary (spectral leakage).
    hann_2d = np.outer(np.hanning(window_size), np.hanning(window_size))

    # Mask of radii in the frequency plane that count as "useful texture".
    cy, cx = window_size // 2, window_size // 2
    Y, X = np.ogrid[:window_size, :window_size]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_min, r_max = 5, cy - 2  # skip the DC area and the outermost ring of noise
    hf_mask = (r >= r_min) & (r <= r_max)

    for i in range(h_steps):
        y_start = i * step_size
        y_end = y_start + window_size
        for j in range(w_steps):
            x_start = j * step_size
            x_end = x_start + window_size

            patch = gray_img[y_start:y_end, x_start:x_end]

            f = np.fft.fft2(patch * hann_2d)
            fshift = np.fft.fftshift(f)  # put DC in the center
            magnitude = np.abs(fshift)

            feature_map[i, j] = np.sum(magnitude[hf_mask])

    # Resize back to the original grid with bicubic interpolation for smoothness.
    feature_map_resized = cv2.resize(feature_map, (W, H), interpolation=cv2.INTER_CUBIC)

    # Min-max scale to [0, 1] so it matches NDVI and saturation when stacked.
    f_min, f_max = feature_map_resized.min(), feature_map_resized.max()
    if f_max > f_min:
        feature_map_resized = (feature_map_resized - f_min) / (f_max - f_min)
    else:
        feature_map_resized = np.zeros_like(feature_map_resized)

    return feature_map_resized


def generate_spectrum_diagnostics(gray_img, output_path):
    """
    Cut three representative 128x128 patches (downtown, park, river) and
    plot their 2D FFT magnitudes side by side. Useful for sanity-checking
    that urban patches really do show stronger grid spikes.
    """
    H, W = gray_img.shape
    patch_size = 128
    hann_2d = np.outer(np.hanning(patch_size), np.hanning(patch_size))

    # Patch coordinates are tuned to the default Warsaw 10x10 km tile.
    urban_patch = gray_img[
        H // 2 - patch_size // 2 : H // 2 + patch_size // 2,
        W // 2 - patch_size // 2 : W // 2 + patch_size // 2,
    ]
    forest_patch = gray_img[100 : 100 + patch_size, 100 : 100 + patch_size]
    water_patch = gray_img[
        H // 2 - patch_size // 2 : H // 2 + patch_size // 2,
        W // 2 + 150 : W // 2 + 150 + patch_size,
    ]

    patches = {
        "Urban grid": urban_patch,
        "Forest canopy": forest_patch,
        "Water surface": water_patch,
    }

    fig, axes = plt.subplots(3, 2, figsize=(10, 14))

    for idx, (name, patch) in enumerate(patches.items()):
        if patch.shape != (patch_size, patch_size):
            patch = cv2.resize(patch, (patch_size, patch_size))

        axes[idx, 0].imshow(patch, cmap="gray")
        axes[idx, 0].set_title(f"Raw patch: {name}")
        axes[idx, 0].axis("off")

        f = np.fft.fft2(patch * hann_2d)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1.0)

        axes[idx, 1].imshow(magnitude, cmap="jet")
        axes[idx, 1].set_title(f"FFT spectrum (log magnitude): {name}")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved FFT diagnostic to {output_path}")


def process_fourier_analysis(year, output_dir="data"):
    """
    Load the aligned cube for `year`, convert it to grayscale, compute the
    sliding-window FFT texture map, and save both the array and a PNG view.
    """
    print(f"\n--- Fourier texture analysis for year {year} ---")
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Aligned raw data not found at {npy_path}. Run alignment.py first.")

    aligned_raw = np.load(npy_path)

    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    gray = cv2.cvtColor((rgb_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    print("Computing sliding-window FFT energy map...")
    fourier_map = compute_local_fft_energy(gray, window_size=64, step_size=8)

    out_npy_path = os.path.join(output_dir, f"fourier_{year}.npy")
    np.save(out_npy_path, fourier_map)
    print(f"Saved Fourier texture map to {out_npy_path}")

    plt.figure(figsize=(8, 8))
    plt.imshow(fourier_map, cmap="inferno")
    plt.title(f"Fourier high-frequency energy - {year}")
    plt.axis("off")
    plt.colorbar(label="Normalized spectral roughness")

    out_png_path = os.path.join(output_dir, f"visual_fourier_{year}.png")
    plt.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Fourier visual to {out_png_path}")

    diag_path = os.path.join(output_dir, f"fourier_diagnostics_{year}.png")
    generate_spectrum_diagnostics(gray, diag_path)

    return fourier_map


if __name__ == "__main__":
    data_dir = "data"
    years = [2018, 2022, 2025]

    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            process_fourier_analysis(yr, data_dir)
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")
