import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_ndvi(aligned_raw):
    """
    Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red).
    Channels in `aligned_raw`: 0=Red, 1=Green, 2=Blue, 3=NIR.
    """
    red = aligned_raw[:, :, 0]
    nir = aligned_raw[:, :, 3]

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

    return ndvi


def compute_saturation(aligned_raw):
    """
    Convert RGB to HSV and return the saturation channel in [0, 1].
    Concrete and asphalt are nearly gray (low saturation); vegetation is
    a vivid green (high saturation), so this gives K-Means an extra cue
    that does not depend on NIR.
    """
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    rgb_uint8 = (rgb_enhanced * 255).astype(np.uint8)

    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1] / 255.0  # back to [0, 1] so it matches NDVI's scale
    return saturation


def apply_clahe_color(aligned_raw):
    """
    Apply CLAHE on the L channel of LAB color space. This boosts local
    contrast (streets, building edges) without shifting the hues.
    """
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    rgb_uint8 = (rgb_enhanced * 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # clipLimit ~3 and an 8x8 tile grid are common defaults: enough local
    # contrast to reveal street layouts without graining out smooth areas.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def process_enhancements(year, output_dir="data"):
    """
    Load the aligned raw cube for `year`, compute NDVI, saturation and a
    CLAHE-enhanced RGB image, save them to disk and produce a 2x2 report PNG.
    """
    print(f"\n--- Spectral and image enhancements for year {year} ---")
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Aligned raw data not found at {npy_path}. Run alignment.py first.")

    aligned_raw = np.load(npy_path)

    ndvi = compute_ndvi(aligned_raw)
    np.save(os.path.join(output_dir, f"ndvi_{year}.npy"), ndvi)
    print(f"Saved NDVI to ndvi_{year}.npy")

    saturation = compute_saturation(aligned_raw)
    np.save(os.path.join(output_dir, f"saturation_{year}.npy"), saturation)
    print(f"Saved saturation to saturation_{year}.npy")

    clahe_rgb = apply_clahe_color(aligned_raw)
    # cv2.imwrite expects BGR; our array is RGB, so swap before saving.
    cv2.imwrite(
        os.path.join(output_dir, f"visual_clahe_{year}.png"),
        cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR),
    )
    print(f"Saved CLAHE image to visual_clahe_{year}.png")

    # 2x2 report: true color | CLAHE | NDVI | saturation.
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    axes[0, 0].imshow(rgb_enhanced)
    axes[0, 0].set_title(f"True color (RGB) - {year}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clahe_rgb)
    axes[0, 1].set_title(f"CLAHE enhanced - {year}")
    axes[0, 1].axis("off")

    # NDVI is clamped to a sensible range so single outlier pixels don't
    # squash the rest of the variation into one color.
    im_ndvi = axes[1, 0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[1, 0].set_title(f"NDVI - {year}")
    axes[1, 0].axis("off")
    fig.colorbar(im_ndvi, ax=axes[1, 0], orientation="horizontal", fraction=0.046, pad=0.04)

    im_sat = axes[1, 1].imshow(saturation, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"HSV saturation - {year}")
    axes[1, 1].axis("off")
    fig.colorbar(im_sat, ax=axes[1, 1], orientation="horizontal", fraction=0.046, pad=0.04)

    png_path = os.path.join(output_dir, f"enhancements_report_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved enhancements report to {png_path}")

    return ndvi, saturation, clahe_rgb


if __name__ == "__main__":
    data_dir = "data"
    years = [2018, 2022, 2025]

    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            process_enhancements(yr, data_dir)
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")
