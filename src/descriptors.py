import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_boundary_fractal_dimension(binary_mask):
    """
    Box-counting fractal dimension of the boundary of `binary_mask`.
    Sprawling, irregular cities give higher D; compact cities give lower D.
    """
    # 1. Extract the boundary: pixels that are in the mask but have at least
    #    one neighbour outside it.
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
    boundary = (binary_mask.astype(np.uint8) - eroded) > 0

    # 2. Pad to the next power of 2 so we can tile it perfectly at every scale.
    H, W = boundary.shape
    p = 2 ** int(np.ceil(np.log2(max(H, W))))
    padded = np.zeros((p, p), dtype=bool)
    padded[:H, :W] = boundary

    # 3. Box sizes 4, 8, 16, ..., p.
    min_box_power = 2  # 2^2 = 4 pixels
    max_box_power = int(np.log2(p))
    powers = np.arange(max_box_power, min_box_power - 1, -1)
    sizes = 2 ** powers

    counts = []
    for s in sizes:
        num_blocks = p // s
        blocks = padded.reshape(num_blocks, s, num_blocks, s)
        active_blocks = np.sum(blocks, axis=(1, 3)) > 0
        counts.append(int(np.sum(active_blocks)))

    counts = np.array(counts)

    # 4. Linear fit of log2(N(s)) vs log2(1/s). The slope is D.
    x = -np.log2(sizes)
    y = np.log2(counts)

    valid = np.where((sizes > 0) & (counts > 0))
    x_valid = x[valid]
    y_valid = y[valid]

    if len(x_valid) < 2:
        return 1.0  # fallback for empty masks

    slope, _ = np.polyfit(x_valid, y_valid, 1)
    return float(slope)


def analyze_urban_features(aligned_raw, binary_urban_mask, year, output_dir="data"):
    """
    Run connected-component analysis on `binary_urban_mask`, compute shape
    descriptors per patch (circularity, solidity) and the global boundary
    fractal dimension. Save a diagnostic plot and return the metrics.
    """
    print(f"\n--- Shape and fractal descriptors for year {year} ---")

    mask_uint8 = (binary_urban_mask * 255).astype(np.uint8)

    # External contours are enough: we treat each blob as a single patch
    # even if it has internal holes.
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    patch_metrics = []
    min_area_filter = 15  # drop tiny speckles (< 1500 m^2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_filter:
            continue

        perimeter = cv2.arcLength(cnt, True)

        # Circularity: 1 for a perfect circle, ~0 for long thin shapes.
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        circularity = min(circularity, 1.0)  # pixel discretization can push it slightly above 1

        # Solidity: how much of the convex hull the shape actually fills.
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        x, y, w, h = cv2.boundingRect(cnt)

        patch_metrics.append(
            {
                "contour": cnt,
                "area_pixels": float(area),
                "area_sq_km": float(area * 10 * 10 / 1e6),  # 10 m x 10 m pixels
                "perimeter_m": float(perimeter * 10),
                "circularity": float(circularity),
                "solidity": float(solidity),
                "centroid": (cx, cy),
                "bbox": (x, y, w, h),
            }
        )

    patch_metrics.sort(key=lambda p: p["area_pixels"], reverse=True)

    fractal_dim = compute_boundary_fractal_dimension(binary_urban_mask)

    print(f"Detected {len(patch_metrics)} urban patches (>= {min_area_filter} px).")
    print(f"Boundary fractal dimension: {fractal_dim:.4f}")
    if patch_metrics:
        print(f"Largest patch: {patch_metrics[0]['area_sq_km']:.2f} km^2")
        avg_circ = np.mean([p["circularity"] for p in patch_metrics])
        avg_sol = np.mean([p["solidity"] for p in patch_metrics])
        print(f"Average circularity: {avg_circ:.3f} | average solidity: {avg_sol:.3f}")

    # Overlay the largest patches on the true-color image as a quick check.
    rgb = aligned_raw[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    overlay_img = (rgb_enhanced * 255).astype(np.uint8).copy()

    render_limit = min(40, len(patch_metrics))
    for p in patch_metrics[:render_limit]:
        x, y, w, h = p["bbox"]
        cx, cy = p["centroid"]
        cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(overlay_img, (cx, cy), 3, (255, 0, 0), -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(overlay_img)
    axes[0].set_title(f"Urban patches and bounding boxes - {year}")
    axes[0].axis("off")

    # Show the boundary that was fed into the fractal counter.
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary_urban_mask.astype(np.uint8), kernel, iterations=1)
    boundary = (binary_urban_mask.astype(np.uint8) - eroded) > 0

    axes[1].imshow(boundary, cmap="plasma")
    axes[1].set_title(f"Urban boundary (D = {fractal_dim:.3f}) - {year}")
    axes[1].axis("off")

    png_path = os.path.join(output_dir, f"descriptors_analysis_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved descriptors report to {png_path}")

    return {
        "num_patches": len(patch_metrics),
        "fractal_dimension": fractal_dim,
        "patches": patch_metrics,
    }


if __name__ == "__main__":
    data_dir = "data"
    years = [2018, 2022, 2025]

    for yr in years:
        aligned_path = os.path.join(data_dir, f"aligned_{yr}.npy")
        temp_veg_path = os.path.join(data_dir, f"temp_cleaned_veg_{yr}.npy")

        if os.path.exists(aligned_path) and os.path.exists(temp_veg_path):
            aligned = np.load(aligned_path)
            # Standalone demo only: use the inverse of the vegetation mask as a urban proxy.
            temp_veg = np.load(temp_veg_path)
            urban_proxy = ~temp_veg
            analyze_urban_features(aligned, urban_proxy, yr, data_dir)
        else:
            print(f"Skip {yr}: data files not found.")
