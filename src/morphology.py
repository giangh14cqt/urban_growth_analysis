import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_kernel(shape="disk", size=3):
    """
    Returns a structuring element of the specified shape and size.
    Supported shapes: 'disk' (ellipse), 'square' (rect), 'cross'.
    """
    if shape == "disk":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == "square":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

def morphological_operation(binary_mask, op_type="open", shape="disk", kernel_size=3):
    """
    Performs basic morphological operations on a binary mask.
    Supported operations: 'open', 'close', 'erode', 'dilate'.
    """
    kernel = get_kernel(shape, kernel_size)
    
    if op_type == "open":
        return cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif op_type == "close":
        return cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    elif op_type == "erode":
        return cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
    elif op_type == "dilate":
        return cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
    else:
        raise ValueError(f"Unsupported morphology operation: {op_type}")

def clean_urban_mask(binary_mask):
    """
    Cleans an urban concrete mask:
    1. Apply MORPH_CLOSE with a disk of size 5 to consolidate building roofs and bridge road fragments.
    2. Apply MORPH_OPEN with a disk of size 3 to remove small speckle noise (cars, street lights, isolated trees).
    """
    mask_closed = morphological_operation(binary_mask, "close", "disk", 5)
    mask_cleaned = morphological_operation(mask_closed, "open", "disk", 3)
    return mask_cleaned

def clean_vegetation_mask(binary_mask):
    """
    Cleans a forest/grass mask:
    1. Apply MORPH_OPEN with a disk of size 3 to erase small road-side trees, cars, or building corners.
    2. Apply MORPH_CLOSE with a disk of size 5 to consolidate larger forest canopy blocks.
    """
    mask_opened = morphological_operation(binary_mask, "open", "disk", 3)
    mask_cleaned = morphological_operation(mask_opened, "close", "disk", 5)
    return mask_cleaned

def clean_water_mask(binary_mask):
    """
    Cleans a water mask:
    1. Apply MORPH_OPEN with a disk of size 3 to remove small shadows or reflections that look like water.
    2. Apply MORPH_CLOSE with a disk of size 7 to fill interior holes and consolidate rivers/lakes.
    """
    mask_opened = morphological_operation(binary_mask, "open", "disk", 3)
    mask_cleaned = morphological_operation(mask_opened, "close", "disk", 7)
    return mask_cleaned

def run_morphology_demo(year, output_dir="data"):
    """
    Loads intermediate data to perform a morphological demo.
    Uses simple thresholds on NDVI to create raw masks, cleans them, and plots before/after.
    """
    print(f"\n--- Running Morphological Processing Demo for Year {year} ---")
    ndvi_path = os.path.join(output_dir, f"ndvi_{year}.npy")
    
    if not os.path.exists(ndvi_path):
        raise FileNotFoundError(f"NDVI data not found at {ndvi_path}. Run enhancement.py first.")
        
    ndvi = np.load(ndvi_path)
    
    # 1. Create a raw vegetation mask based on standard NDVI threshold (> 0.45)
    raw_veg = ndvi > 0.45
    cleaned_veg = clean_vegetation_mask(raw_veg)
    
    # 2. Save cleaned vegetation array just as a placeholder for segmentation validation
    np.save(os.path.join(output_dir, f"temp_cleaned_veg_{year}.npy"), cleaned_veg)
    
    # 3. Create a raw non-vegetated/water mask based on NDVI threshold (< 0.15)
    # We will use this to showcase how morphology bridges urban elements
    raw_urban_proxy = ndvi < 0.15
    cleaned_urban_proxy = clean_urban_mask(raw_urban_proxy)
    
    # 4. Generate Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw Vegetation
    axes[0, 0].imshow(raw_veg, cmap="Greens")
    axes[0, 0].set_title(f"Raw Vegetation Mask (NDVI > 0.45) - {year}")
    axes[0, 0].axis("off")
    
    # Cleaned Vegetation
    axes[0, 1].imshow(cleaned_veg, cmap="Greens")
    axes[0, 1].set_title(f"Cleaned Vegetation Mask (Opening + Closing) - {year}")
    axes[0, 1].axis("off")
    
    # Raw Urban Proxy
    axes[1, 0].imshow(raw_urban_proxy, cmap="gray")
    axes[1, 0].set_title(f"Raw Urban/Concrete Mask (NDVI < 0.15) - {year}")
    axes[1, 0].axis("off")
    
    # Cleaned Urban Proxy
    axes[1, 1].imshow(cleaned_urban_proxy, cmap="gray")
    axes[1, 1].set_title(f"Cleaned Urban Mask (Closing + Opening) - {year}")
    axes[1, 1].axis("off")
    
    png_path = os.path.join(output_dir, f"morphology_demo_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved morphological cleaning visual report to {png_path}")
    
    return cleaned_veg, cleaned_urban_proxy

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    years = [2018, 2022, 2025]
    
    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"ndvi_{yr}.npy")):
            run_morphology_demo(yr, data_dir)
        else:
            print(f"Skip {yr}: ndvi_{yr}.npy not found.")
