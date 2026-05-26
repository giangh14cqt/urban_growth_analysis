import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_kernel(shape="disk", size=3):
    """
    Return an OpenCV structuring element.
    Supported shapes: 'disk' (ellipse), 'square' (rect), 'cross'.
    """
    if shape == "square":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    # default: disk / ellipse, which is rotation-symmetric
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morphological_operation(binary_mask, op_type="open", shape="disk", kernel_size=3):
    """Apply a single morphological operation: 'open', 'close', 'erode' or 'dilate'."""
    kernel = get_kernel(shape, kernel_size)
    mask = binary_mask.astype(np.uint8)

    if op_type == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if op_type == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if op_type == "erode":
        return cv2.erode(mask, kernel, iterations=1)
    if op_type == "dilate":
        return cv2.dilate(mask, kernel, iterations=1)
    raise ValueError(f"Unsupported morphology operation: {op_type}")


def clean_urban_mask(binary_mask):
    """
    Urban mask cleanup:
      1. Close with a disk of size 5 to bridge roofs and roads into solid blocks.
      2. Open with a disk of size 3 to remove leftover speckles (cars, lamps, lone trees).
    """
    mask_closed = morphological_operation(binary_mask, "close", "disk", 5)
    return morphological_operation(mask_closed, "open", "disk", 3)


def clean_vegetation_mask(binary_mask):
    """
    Vegetation mask cleanup:
      1. Open with a disk of size 3 to drop roadside trees and isolated bushes.
      2. Close with a disk of size 5 to fill small gaps inside the canopy.
    """
    mask_opened = morphological_operation(binary_mask, "open", "disk", 3)
    return morphological_operation(mask_opened, "close", "disk", 5)


def clean_water_mask(binary_mask):
    """
    Water mask cleanup:
      1. Open with a disk of size 3 to remove dark shadows mislabelled as water.
      2. Close with a disk of size 7 to fill interior gaps in rivers/lakes.
    """
    mask_opened = morphological_operation(binary_mask, "open", "disk", 3)
    return morphological_operation(mask_opened, "close", "disk", 7)


def run_morphology_demo(year, output_dir="data"):
    """
    Standalone demo: build crude NDVI-threshold masks for vegetation and
    non-vegetation, run the cleanup routines, and save a before/after plot.
    """
    print(f"\n--- Morphology demo for year {year} ---")
    ndvi_path = os.path.join(output_dir, f"ndvi_{year}.npy")

    if not os.path.exists(ndvi_path):
        raise FileNotFoundError(f"NDVI data not found at {ndvi_path}. Run enhancement.py first.")

    ndvi = np.load(ndvi_path)

    raw_veg = ndvi > 0.45
    cleaned_veg = clean_vegetation_mask(raw_veg)
    np.save(os.path.join(output_dir, f"temp_cleaned_veg_{year}.npy"), cleaned_veg)

    raw_urban_proxy = ndvi < 0.15
    cleaned_urban_proxy = clean_urban_mask(raw_urban_proxy)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(raw_veg, cmap="Greens")
    axes[0, 0].set_title(f"Raw vegetation mask (NDVI > 0.45) - {year}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cleaned_veg, cmap="Greens")
    axes[0, 1].set_title(f"Cleaned vegetation (open + close) - {year}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(raw_urban_proxy, cmap="gray")
    axes[1, 0].set_title(f"Raw urban proxy (NDVI < 0.15) - {year}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cleaned_urban_proxy, cmap="gray")
    axes[1, 1].set_title(f"Cleaned urban (close + open) - {year}")
    axes[1, 1].axis("off")

    png_path = os.path.join(output_dir, f"morphology_demo_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved morphology demo to {png_path}")

    return cleaned_veg, cleaned_urban_proxy


if __name__ == "__main__":
    data_dir = "data"
    years = [2018, 2022, 2025]

    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"ndvi_{yr}.npy")):
            run_morphology_demo(yr, data_dir)
        else:
            print(f"Skip {yr}: ndvi_{yr}.npy not found.")
