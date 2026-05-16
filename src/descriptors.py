import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_boundary_fractal_dimension(binary_mask):
    """
    Computes the Fractal Dimension of the urban boundary using the Box-Counting algorithm.
    Sprawling, irregular cities have a higher fractal dimension, while compact ones are lower.
    """
    # 1. Extract the urban boundary (Boundary = Original - Eroded)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
    boundary = (binary_mask.astype(np.uint8) - eroded) > 0
    
    # 2. Pad boundary array to the nearest power of 2
    H, W = boundary.shape
    max_dim = max(H, W)
    p = 2 ** int(np.ceil(np.log2(max_dim)))
    padded = np.zeros((p, p), dtype=bool)
    padded[:H, :W] = boundary
    
    # 3. Successive division by powers of 2 (box sizing)
    # Range of box sizes: from p down to 4 pixels
    min_box_power = 2  # 2^2 = 4 pixels
    max_box_power = int(np.log2(p))
    powers = np.arange(max_box_power, min_box_power - 1, -1)
    sizes = 2 ** powers
    
    counts = []
    for s in sizes:
        # Reshape padded array to group into s x s blocks
        num_blocks = p // s
        blocks = padded.reshape(num_blocks, s, num_blocks, s)
        # Check if each s x s block has at least one active boundary pixel
        active_blocks = np.sum(blocks, axis=(1, 3)) > 0
        counts.append(np.sum(active_blocks))
        
    counts = np.array(counts)
    
    # 4. Perform log-log linear regression: log(N(s)) = D * log(1/s) + C
    # -> log2(counts) = -D * log2(sizes) + C
    # Let x = -log2(sizes) and y = log2(counts)
    x = -np.log2(sizes)
    y = np.log2(counts)
    
    # Filter out empty sizes (if any) to prevent log(0)
    valid_idx = np.where((sizes > 0) & (counts > 0))
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    if len(x_valid) < 2:
        return 1.0  # Fallback for empty masks
        
    # Linear fit
    coeffs = np.polyfit(x_valid, y_valid, 1)
    fractal_dim = coeffs[0]
    
    return float(fractal_dim)

def analyze_urban_features(aligned_raw, binary_urban_mask, year, output_dir="data"):
    """
    Performs Connected Component Labeling on the urban mask, extracts shape descriptors
    (Circularity, Solidity) for each patch, and calculates the overall boundary fractal dimension.
    """
    print(f"\n--- Extracting Shape & Fractal Feature Descriptors for Year {year} ---")
    
    mask_uint8 = (binary_urban_mask * 255).astype(np.uint8)
    
    # 1. Extract contours to calculate Solidity and Circularity via shape parameters
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patch_metrics = []
    min_area_filter = 15  # Filter out tiny speckles (noise < 150 sq meters)
    
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area_filter:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        
        # Compute Circularity: 4 * pi * Area / (Perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        circularity = min(circularity, 1.0)  # Cap at 1.0 due to discrete pixel boundaries
        
        # Compute Solidity: Area / Convex Hull Area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        
        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        patch_metrics.append({
            "contour": cnt,
            "area_pixels": float(area),
            "area_sq_km": float(area * 10 * 10 / 1e6), # 10m x 10m pixels
            "perimeter_m": float(perimeter * 10),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "centroid": (cx, cy),
            "bbox": (x, y, w, h)
        })
        
    # Sort patches by size to isolate urban cores
    patch_metrics = sorted(patch_metrics, key=lambda x: x["area_pixels"], reverse=True)
    
    # 2. Calculate boundary fractal dimension
    fractal_dim = compute_boundary_fractal_dimension(binary_urban_mask)
    
    print(f"Detected {len(patch_metrics)} discrete urban sprawl patches (area >= {min_area_filter} px).")
    print(f"Global Boundary Fractal Dimension: {fractal_dim:.4f}")
    if len(patch_metrics) > 0:
        print(f"Largest Core Area: {patch_metrics[0]['area_sq_km']:.2f} sq km")
        avg_circ = np.mean([p["circularity"] for p in patch_metrics])
        avg_sol = np.mean([p["solidity"] for p in patch_metrics])
        print(f"Average Circularity: {avg_circ:.3f} | Average Solidity: {avg_sol:.3f}")
    
    # 3. Create sprawl pockets overlay map
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    
    # Draw on overlay
    overlay_img = (rgb_enhanced * 255).astype(np.uint8).copy()
    
    # Render up to the top 40 largest patches
    render_limit = min(40, len(patch_metrics))
    for p in patch_metrics[:render_limit]:
        x, y, w, h = p["bbox"]
        cx, cy = p["centroid"]
        
        # Green bounding box for sprawl pocket
        cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Red center point
        cv2.circle(overlay_img, (cx, cy), 3, (255, 0, 0), -1)
        
    # Generate diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(overlay_img)
    axes[0].set_title(f"Urban Sprawl Patches & BBoxes Overlay - {year}")
    axes[0].axis("off")
    
    # Display the eroded boundary that was used to calculate the Fractal Dimension
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_urban_mask.astype(np.uint8), kernel, iterations=1)
    boundary = (binary_urban_mask.astype(np.uint8) - eroded) > 0
    
    axes[1].imshow(boundary, cmap="plasma")
    axes[1].set_title(f"Urban Fringe Boundary (Fractal D: {fractal_dim:.3f}) - {year}")
    axes[1].axis("off")
    
    png_path = os.path.join(output_dir, f"descriptors_analysis_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved feature descriptors visual report to {png_path}")
    
    return {
        "num_patches": len(patch_metrics),
        "fractal_dimension": fractal_dim,
        "patches": patch_metrics
    }

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    years = [2018, 2022, 2025]
    
    for yr in years:
        aligned_path = os.path.join(data_dir, f"aligned_{yr}.npy")
        temp_veg_path = os.path.join(data_dir, f"temp_cleaned_veg_{yr}.npy")
        
        if os.path.exists(aligned_path) and os.path.exists(temp_veg_path):
            aligned = np.load(aligned_path)
            # We use the INVERSE of the vegetation mask as an urban concrete proxy for this test demo
            temp_veg = np.load(temp_veg_path)
            urban_proxy = ~temp_veg
            
            analyze_urban_features(aligned, urban_proxy, yr, data_dir)
        else:
            print(f"Skip {yr}: data files not found.")
