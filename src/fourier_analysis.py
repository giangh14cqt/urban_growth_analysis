import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_local_fft_energy(gray_img, window_size=64, step_size=8):
    """
    Computes local 2D Fourier high-frequency energy using a sliding window.
    High-frequency energy is a strong proxy for structural complexity (buildings/roads).
    """
    H, W = gray_img.shape
    h_steps = (H - window_size) // step_size + 1
    w_steps = (W - window_size) // step_size + 1
    
    feature_map = np.zeros((h_steps, w_steps))
    
    # 2D Hanning Window to prevent boundary spectral leakage
    hann_y = np.hanning(window_size)
    hann_x = np.hanning(window_size)
    hann_2d = np.outer(hann_y, hann_x)
    
    # Precompute radial distance mask
    cy, cx = window_size // 2, window_size // 2
    Y, X = np.ogrid[:window_size, :window_size]
    r = np.sqrt((Y - cy)**2 + (X - cx)**2)
    
    # Extract frequencies in a medium-to-high radial band
    # Ignoring low frequency (0-4 px) and extremely high frequency noise (> cy-2)
    r_min, r_max = 5, cy - 2
    hf_mask = (r >= r_min) & (r <= r_max)
    
    for i in range(h_steps):
        y_start = i * step_size
        y_end = y_start + window_size
        for j in range(w_steps):
            x_start = j * step_size
            x_end = x_start + window_size
            
            patch = gray_img[y_start:y_end, x_start:x_end]
            
            # Apply window and run 2D FFT
            patch_windowed = patch * hann_2d
            f = np.fft.fft2(patch_windowed)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Sum energy in the specified frequency band
            hf_energy = np.sum(magnitude[hf_mask])
            feature_map[i, j] = hf_energy
            
    # Resize map back to native resolution using bicubic interpolation for smoothness
    feature_map_resized = cv2.resize(feature_map, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Normalize features to [0.0, 1.0] range
    f_min, f_max = feature_map_resized.min(), feature_map_resized.max()
    if f_max > f_min:
        feature_map_resized = (feature_map_resized - f_min) / (f_max - f_min)
    else:
        feature_map_resized = np.zeros_like(feature_map_resized)
        
    return feature_map_resized

def generate_spectrum_diagnostics(gray_img, output_path):
    """
    Extracts representative patches for Urban, Forest, and Water
    and plots their 2D Fourier Magnitude spectra to demonstrate grid spikes.
    """
    H, W = gray_img.shape
    patch_size = 128
    hann_2d = np.outer(np.hanning(patch_size), np.hanning(patch_size))
    
    # Locate characteristic coordinates for Warsaw 10x10 km centered at (52.2297, 21.0122)
    # Urban Center: Downtown Warsaw (~center)
    urban_patch = gray_img[H//2 - patch_size//2 : H//2 + patch_size//2, 
                           W//2 - patch_size//2 : W//2 + patch_size//2]
                           
    # Forest Patch: Lazienki Park or Kampinos fringe (usually top-left / bottom-right)
    # Let's crop a patch from top-left (often vegetation)
    forest_patch = gray_img[100 : 100 + patch_size, 100 : 100 + patch_size]
    
    # Water Patch: Vistula River (crosses vertically around W//2 + 100)
    # Let's crop from where the river is highly likely
    water_patch = gray_img[H//2 - patch_size//2 : H//2 + patch_size//2, 
                           W//2 + 150 : W//2 + 150 + patch_size]
                           
    patches = {"Urban Grid": urban_patch, "Forest Canopy": forest_patch, "Water Surface": water_patch}
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    
    for idx, (name, patch) in enumerate(patches.items()):
        # Handle shape mismatch just in case
        if patch.shape != (patch_size, patch_size):
            patch = cv2.resize(patch, (patch_size, patch_size))
            
        # 1. Plot raw patch
        axes[idx, 0].imshow(patch, cmap="gray")
        axes[idx, 0].set_title(f"Raw Patch: {name}")
        axes[idx, 0].axis("off")
        
        # 2. Compute FFT
        f = np.fft.fft2(patch * hann_2d)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1.0)  # Log scale for visualization
        
        axes[idx, 1].imshow(magnitude, cmap="jet")
        axes[idx, 1].set_title(f"FFT Spectrum (Log Magnitude): {name}")
        axes[idx, 1].axis("off")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Fourier Spectrum comparison diagnostic to {output_path}")

def process_fourier_analysis(year, output_dir="data"):
    """
    Loads aligned raw data, extracts grayscale reference,
    runs local sliding-window FFT, and saves analytical and diagnostic outputs.
    """
    print(f"\n--- Running Fourier Spatial Frequency Analysis for Year {year} ---")
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Aligned raw data file not found at {npy_path}. Run alignment.py first.")
        
    aligned_raw = np.load(npy_path)
    
    # Extract RGB visual bands
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    # Convert to grayscale for structural analysis
    gray = cv2.cvtColor((rgb_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 1. Compute sliding window FFT high frequency energy map
    print("Calculating local 2D FFT sliding window energy map (this may take a few seconds)...")
    fourier_map = compute_local_fft_energy(gray, window_size=64, step_size=8)
    
    # Save the feature map
    out_npy_path = os.path.join(output_dir, f"fourier_{year}.npy")
    np.save(out_npy_path, fourier_map)
    print(f"Saved localized Fourier Texture Map to {out_npy_path}")
    
    # 2. Save visual representation
    plt.figure(figsize=(8, 8))
    plt.imshow(fourier_map, cmap="inferno")
    plt.title(f"Fourier Local High-Frequency Energy Map - {year}")
    plt.axis("off")
    plt.colorbar(label="Normalized Spectral Roughness")
    
    out_png_path = os.path.join(output_dir, f"visual_fourier_{year}.png")
    plt.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Fourier visual map to {out_png_path}")
    
    # 3. Generate spectrum comparison reports (only once or for all years)
    diag_path = os.path.join(output_dir, f"fourier_diagnostics_{year}.png")
    generate_spectrum_diagnostics(gray, diag_path)
    
    return fourier_map

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    years = [2018, 2022, 2025]
    
    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            process_fourier_analysis(yr, data_dir)
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")
