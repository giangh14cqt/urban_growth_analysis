import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_gray_reference(raw_array):
    """
    Build a contrast-stretched grayscale image from the raw float32 cube.
    Uses the visible RGB bands (indices 0, 1, 2).
    """
    rgb = raw_array[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    gray = cv2.cvtColor((rgb_enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray


def register_and_align(base_raw, target_raw, year, output_dir="data"):
    """
    Align `target_raw` to `base_raw` with ORB feature matching and a
    RANSAC homography. All 5 bands are warped with the same matrix so
    the spectral relationships stay intact.
    """
    print(f"\n--- Aligning satellite image for year {year} to 2018 base ---")

    gray_base = extract_gray_reference(base_raw)
    gray_target = extract_gray_reference(target_raw)

    # ORB finds corner-like keypoints and gives each one a binary descriptor.
    # 3000 keypoints is enough to get a stable homography on 1000x1000 tiles.
    orb = cv2.ORB_create(nfeatures=3000)
    kp_base, des_base = orb.detectAndCompute(gray_base, None)
    kp_target, des_target = orb.detectAndCompute(gray_target, None)

    if des_base is None or des_target is None:
        print("Warning: feature description failed. Returning the unaligned image.")
        return target_raw

    # Brute-force KNN matching with Hamming distance (the natural metric
    # for binary descriptors). Two nearest matches per keypoint, then
    # Lowe's ratio test to drop ambiguous ones.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_base, des_target, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    print(f"Keypoints: {len(kp_base)} in base, {len(kp_target)} in {year}.")
    print(f"Kept {len(good_matches)} matches after Lowe's ratio test.")

    # Estimate the homography H that maps the target frame onto the base.
    # RANSAC ignores outlier matches automatically.
    min_matches = 15
    if len(good_matches) >= min_matches:
        src_pts = np.float32([kp_base[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("Warning: homography estimation failed. Using identity transform.")
            H = np.eye(3)
        else:
            inliers = int(np.sum(mask))
            print(f"RANSAC homography: {inliers} inliers ({inliers / len(good_matches) * 100:.1f}%).")
    else:
        print(f"Warning: only {len(good_matches)} matches (< {min_matches}). Using identity transform.")
        H = np.eye(3)

    # Warp every band with the same H so NDVI etc. stays meaningful.
    # Bilinear interpolation is the standard choice for continuous reflectance.
    H_rows, W_cols = base_raw.shape[:2]
    aligned_raw = np.zeros_like(target_raw)

    for c in range(target_raw.shape[2]):
        aligned_raw[:, :, c] = cv2.warpPerspective(
            target_raw[:, :, c],
            H,
            (W_cols, H_rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"aligned_{year}.npy")
    np.save(npy_path, aligned_raw)
    print(f"Saved aligned multi-spectral array to {npy_path}")

    # Diagnostic plot: red/cyan overlay + absolute difference.
    gray_aligned = extract_gray_reference(aligned_raw)

    overlay = np.zeros((H_rows, W_cols, 3), dtype=np.uint8)
    overlay[:, :, 0] = gray_base       # red channel: base year
    overlay[:, :, 1] = gray_aligned    # green + blue channel: aligned target -> cyan
    overlay[:, :, 2] = gray_aligned

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(overlay)
    axes[0].set_title(f"Alignment overlay (red: 2018 | cyan: {year})")
    axes[0].axis("off")

    diff_image = cv2.absdiff(gray_base, gray_aligned)
    axes[1].imshow(diff_image, cmap="hot")
    axes[1].set_title(f"Absolute difference (2018 vs aligned {year})")
    axes[1].axis("off")

    png_path = os.path.join(output_dir, f"alignment_check_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved registration report to {png_path}")

    return aligned_raw


if __name__ == "__main__":
    data_dir = "data"
    raw_2018_path = os.path.join(data_dir, "raw_2018.npy")
    raw_2022_path = os.path.join(data_dir, "raw_2022.npy")
    raw_2025_path = os.path.join(data_dir, "raw_2025.npy")

    if not (os.path.exists(raw_2018_path) and os.path.exists(raw_2022_path) and os.path.exists(raw_2025_path)):
        print("Error: raw data files not found. Run the Sentinel fetcher first.")
    else:
        base_18 = np.load(raw_2018_path)
        raw_22 = np.load(raw_2022_path)
        raw_25 = np.load(raw_2025_path)

        register_and_align(base_18, raw_22, 2022)
        register_and_align(base_18, raw_25, 2025)

        # 2018 is the reference, so copy it over to keep a uniform filename pattern.
        np.save(os.path.join(data_dir, "aligned_2018.npy"), base_18)
        print("Copied 2018 base data to aligned_2018.npy")
