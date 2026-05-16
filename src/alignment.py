import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_gray_reference(raw_array):
    """
    Extracts a high-contrast grayscale image from the FLOAT32 raw Sentinel-2 array.
    Using B04 (Red), B03 (Green), and B02 (Blue) channels.
    """
    # Extract RGB bands (indices 0, 1, 2)
    rgb = raw_array[:, :, :3]
    # Enhance contrast using standard visual gain
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    # Convert to uint8 grayscale
    gray = cv2.cvtColor((rgb_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray

def register_and_align(base_raw, target_raw, year, output_dir="data"):
    """
    Aligns the target_raw array to the base_raw array using ORB feature matching and RANSAC.
    Warps all 5 channels of the target array and saves both raw and diagnostic outputs.
    """
    print(f"\n--- Aligning Satellite Image for Year {year} to 2018 Base ---")
    
    # 1. Extract grayscale reference images
    gray_base = extract_gray_reference(base_raw)
    gray_target = extract_gray_reference(target_raw)
    
    # 2. Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=3000)
    
    # Find keypoints and descriptors
    kp_base, des_base = orb.detectAndCompute(gray_base, None)
    kp_target, des_target = orb.detectAndCompute(gray_target, None)
    
    if des_base is None or des_target is None:
        print("Warning: Feature description failed. Falling back to unaligned original.")
        return target_raw
        
    # 3. Match descriptors using Brute-Force KNN Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_base, des_target, k=2)
    
    # Apply Lowe's Ratio Test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    print(f"Detected {len(kp_base)} keypoints in 2018, {len(kp_target)} in {year}.")
    print(f"Found {len(good_matches)} high-quality matches after Lowe's ratio test.")
    
    # 4. Homography Estimation using RANSAC
    min_matches = 15
    if len(good_matches) >= min_matches:
        # Extract matching keypoint coordinates
        src_pts = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography H: maps points in target (dst_pts) to base (src_pts)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Warning: Homography estimation failed. Using unaligned data.")
            H = np.eye(3)
        else:
            inliers = np.sum(mask)
            print(f"RANSAC Homography estimated with {inliers} inliers ({inliers/len(good_matches)*100:.1f}%).")
    else:
        print(f"Warning: Insufficient matches ({len(good_matches)} < {min_matches}). Using unaligned data.")
        H = np.eye(3)
        
    # 5. Warp all 5 FLOAT32 channels using cv2.warpPerspective
    H_rows, W_cols = base_raw.shape[:2]
    aligned_raw = np.zeros_like(target_raw)
    
    for c in range(target_raw.shape[2]):
        aligned_raw[:, :, c] = cv2.warpPerspective(
            target_raw[:, :, c], 
            H, 
            (W_cols, H_rows), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        
    # 6. Save the registered raw multi-spectral data
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")
    np.save(npy_path, aligned_raw)
    print(f"Saved aligned multi-spectral array to {npy_path}")
    
    # 7. Generate diagnostic alignment check plot
    # Compute red-cyan overlay of aligned images
    gray_aligned = extract_gray_reference(aligned_raw)
    
    overlay = np.zeros((H_rows, W_cols, 3), dtype=np.uint8)
    overlay[:, :, 0] = gray_base      # Red channel: 2018 base
    overlay[:, :, 1] = gray_aligned   # Green channel: Aligned target
    overlay[:, :, 2] = gray_aligned   # Blue channel: Aligned target (creates Cyan)
    
    # Red-cyan checkboard or side-by-side diagnostic visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(overlay)
    axes[0].set_title(f"Alignment Overlay (Red: 2018 | Cyan: {year})")
    axes[0].axis("off")
    
    # Show difference image to highlight changes or offsets
    diff_image = cv2.absdiff(gray_base, gray_aligned)
    axes[1].imshow(diff_image, cmap="hot")
    axes[1].set_title(f"Absolute Visual Difference (Base 2018 vs Aligned {year})")
    axes[1].axis("off")
    
    png_path = os.path.join(output_dir, f"alignment_check_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved registration visual report to {png_path}")
    
    return aligned_raw

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    raw_2018_path = os.path.join(data_dir, "raw_2018.npy")
    raw_2022_path = os.path.join(data_dir, "raw_2022.npy")
    raw_2025_path = os.path.join(data_dir, "raw_2025.npy")
    
    if not (os.path.exists(raw_2018_path) and os.path.exists(raw_2022_path) and os.path.exists(raw_2025_path)):
        print("Error: Raw data files not found. Run Sentinel API fetcher first.")
    else:
        # Load raw data
        base_18 = np.load(raw_2018_path)
        raw_22 = np.load(raw_2022_path)
        raw_25 = np.load(raw_2025_path)
        
        # Execute registration
        register_and_align(base_18, raw_22, 2022)
        register_and_align(base_18, raw_25, 2025)
        
        # 2018 is already aligned with itself, let's save aligned_2018.npy for API uniformity
        np.save(os.path.join(data_dir, "aligned_2018.npy"), base_18)
        print("Uniformity: Copied 2018 base data to aligned_2018.npy")
