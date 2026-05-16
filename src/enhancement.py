import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_ndvi(aligned_raw):
    """
    Computes the Normalized Difference Vegetation Index (NDVI).
    Formula: (NIR - Red) / (NIR + Red) -> (B08 - B04) / (B08 + B04)
    Channels in aligned_raw: 0: Red, 1: Green, 2: Blue, 3: NIR
    """
    red = aligned_raw[:, :, 0]
    nir = aligned_raw[:, :, 3]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        # Handle boundary conditions (division by zero or invalid negative values)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        
    return ndvi

def compute_saturation(aligned_raw):
    """
    Converts aligned raw to RGB visual, transforms to HSV,
    and returns the normalized Saturation channel [0.0, 1.0].
    Built-up/urban concrete has low saturation, while vegetation has high saturation.
    """
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    rgb_uint8 = (rgb_enhanced * 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1] / 255.0  # Normalize to [0.0, 1.0]
    return saturation

def apply_clahe_color(aligned_raw):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    on the L (Luminance) channel of the LAB color space to enhance spatial detail
    without altering original spectral hues.
    """
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    rgb_uint8 = (rgb_enhanced * 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Re-merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def process_enhancements(year, output_dir="data"):
    """
    Loads aligned_raw data for a target year, processes NDVI, Saturation, and CLAHE,
    saves the analytical numpy arrays and outputs beautiful visual reports.
    """
    print(f"\n--- Processing Spectral & Image Enhancements for Year {year} ---")
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Aligned raw data file not found at {npy_path}. Run alignment.py first.")
        
    aligned_raw = np.load(npy_path)
    
    # 1. Compute NDVI
    ndvi = compute_ndvi(aligned_raw)
    np.save(os.path.join(output_dir, f"ndvi_{year}.npy"), ndvi)
    print(f"Calculated and saved NDVI to ndvi_{year}.npy")
    
    # 2. Compute Saturation
    saturation = compute_saturation(aligned_raw)
    np.save(os.path.join(output_dir, f"saturation_{year}.npy"), saturation)
    print(f"Calculated and saved Saturation to saturation_{year}.npy")
    
    # 3. Compute CLAHE
    clahe_rgb = apply_clahe_color(aligned_raw)
    # Save visual RGB with CLAHE
    clahe_pil = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR)  # Convert for saving via OpenCV if needed
    cv2.imwrite(os.path.join(output_dir, f"visual_clahe_{year}.png"), clahe_pil)
    print(f"Calculated and saved CLAHE enhanced image to visual_clahe_{year}.png")
    
    # 4. Generate Diagnostic Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # True Color
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    axes[0, 0].imshow(rgb_enhanced)
    axes[0, 0].set_title(f"True Color Visual (RGB) - {year}")
    axes[0, 0].axis("off")
    
    # CLAHE Enhanced
    axes[0, 1].imshow(clahe_rgb)
    axes[0, 1].set_title(f"CLAHE Enhanced Visual - {year}")
    axes[0, 1].axis("off")
    
    # NDVI (Viridis Colormap)
    im_ndvi = axes[1, 0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[1, 0].set_title(f"NDVI Vegetation Index - {year}")
    axes[1, 0].axis("off")
    fig.colorbar(im_ndvi, ax=axes[1, 0], orientation="horizontal", fraction=0.046, pad=0.04)
    
    # Saturation (Magma Colormap)
    im_sat = axes[1, 1].imshow(saturation, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"HSV Saturation Channel - {year}")
    axes[1, 1].axis("off")
    fig.colorbar(im_sat, ax=axes[1, 1], orientation="horizontal", fraction=0.046, pad=0.04)
    
    png_path = os.path.join(output_dir, f"enhancements_report_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved spectral enhancement visual report to {png_path}")
    
    return ndvi, saturation, clahe_rgb

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    years = [2018, 2022, 2025]
    
    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            process_enhancements(yr, data_dir)
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")
