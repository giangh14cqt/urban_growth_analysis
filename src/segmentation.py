import os

import cv2  # noqa: F401  (kept for downstream notebooks that import from this module)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from src.descriptors import analyze_urban_features
from src.morphology import clean_urban_mask, clean_vegetation_mask, clean_water_mask


def run_multi_spectral_segmentation(year, data_dir="data"):
    """
    Stack spectral bands and engineered features, cluster them with K-Means,
    assign cluster IDs to land-cover classes from the centroid values, clean
    the masks with morphology, and report area statistics.
    """
    print(f"\n--- Multi-spectral segmentation for year {year} ---")

    aligned = np.load(os.path.join(data_dir, f"aligned_{year}.npy"))
    ndvi = np.load(os.path.join(data_dir, f"ndvi_{year}.npy"))
    fourier = np.load(os.path.join(data_dir, f"fourier_{year}.npy"))
    saturation = np.load(os.path.join(data_dir, f"saturation_{year}.npy"))

    H, W, _ = aligned.shape

    red = aligned[:, :, 0]
    green = aligned[:, :, 1]
    blue = aligned[:, :, 2]
    nir = aligned[:, :, 3]

    # Per-pixel feature vector: [R, G, B, NIR, NDVI, Fourier texture, saturation].
    features = np.stack(
        [
            red.ravel(),
            green.ravel(),
            blue.ravel(),
            nir.ravel(),
            ndvi.ravel(),
            fourier.ravel(),
            saturation.ravel(),
        ],
        axis=1,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

    print("Fitting K-Means on the pixel stack...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    raw_segmentation = labels.reshape((H, W))

    # K-Means returns cluster IDs in a random order. Re-label them using
    # the centroid coordinates so the mapping is stable across years.
    centers = kmeans.cluster_centers_  # shape (3, 7)
    forest_idx = int(np.argmax(centers[:, 4]))  # highest mean NDVI -> vegetation
    remaining = [i for i in range(3) if i != forest_idx]
    water_idx = remaining[int(np.argmin(centers[remaining, 3]))]  # lowest NIR -> water
    urban_idx = [i for i in range(3) if i not in (forest_idx, water_idx)][0]

    print(f"Cluster mapping -> forest: {forest_idx} | water: {water_idx} | urban: {urban_idx}")

    raw_urban = raw_segmentation == urban_idx
    raw_forest = raw_segmentation == forest_idx
    raw_water = raw_segmentation == water_idx

    print("Cleaning masks with morphology...")
    cleaned_urban = clean_urban_mask(raw_urban)
    cleaned_forest = clean_vegetation_mask(raw_forest)
    cleaned_water = clean_water_mask(raw_water)

    # Morphology can grow masks slightly; resolve overlaps by priority.
    final_segmentation = np.zeros((H, W), dtype=np.uint8)
    final_segmentation[cleaned_urban > 0] = 1   # urban
    final_segmentation[cleaned_forest > 0] = 2  # forest/grass
    final_segmentation[cleaned_water > 0] = 3   # water

    np.save(os.path.join(data_dir, f"classified_{year}.npy"), final_segmentation)
    print(f"Saved classified array to classified_{year}.npy")

    total_pixels = H * W
    urban_pixels = int(np.sum(final_segmentation == 1))
    forest_pixels = int(np.sum(final_segmentation == 2))
    water_pixels = int(np.sum(final_segmentation == 3))

    # 1 pixel = 10 m x 10 m = 1e-4 km^2
    px_to_sq_km = 0.0001

    urban_area_km2 = urban_pixels * px_to_sq_km
    forest_area_km2 = forest_pixels * px_to_sq_km
    water_area_km2 = water_pixels * px_to_sq_km

    urban_pct = urban_pixels / total_pixels * 100
    forest_pct = forest_pixels / total_pixels * 100
    water_pct = water_pixels / total_pixels * 100

    print("\nLand cover statistics:")
    print(f"  Urban:  {urban_area_km2:.2f} km^2 ({urban_pct:.2f}%)")
    print(f"  Forest: {forest_area_km2:.2f} km^2 ({forest_pct:.2f}%)")
    print(f"  Water:  {water_area_km2:.2f} km^2 ({water_pct:.2f}%)")

    descriptors = analyze_urban_features(aligned, cleaned_urban, year, data_dir)

    # Side-by-side report: true color and classified map.
    cmap_custom = plt.matplotlib.colors.ListedColormap(["#dcdcdc", "#555555", "#2ca02c", "#1f77b4"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    rgb = aligned[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    axes[0].imshow(rgb_enhanced)
    axes[0].set_title(f"True color - {year}")
    axes[0].axis("off")

    axes[1].imshow(final_segmentation, cmap=cmap_custom, vmin=0, vmax=3)
    axes[1].set_title(f"Land cover classification - {year}")
    axes[1].axis("off")

    legend_colors = ["#555555", "#2ca02c", "#1f77b4"]
    legend_labels = ["Urban (concrete/roads)", "Forest and grass", "Water"]
    patches = [
        plt.matplotlib.patches.Patch(color=legend_colors[i], label=legend_labels[i])
        for i in range(len(legend_colors))
    ]
    plt.legend(handles=patches, loc="lower right", borderaxespad=1.0)

    png_path = os.path.join(data_dir, f"segmentation_report_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved segmentation report to {png_path}")

    return {
        "year": year,
        "urban_km2": urban_area_km2,
        "urban_pct": urban_pct,
        "forest_km2": forest_area_km2,
        "forest_pct": forest_pct,
        "water_km2": water_area_km2,
        "water_pct": water_pct,
        "fractal_dimension": descriptors["fractal_dimension"],
        "num_urban_patches": descriptors["num_patches"],
    }


if __name__ == "__main__":
    data_dir = "data"
    years = [2018, 2022, 2025]

    results = {}
    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            results[yr] = run_multi_spectral_segmentation(yr, data_dir)
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")

    if len(results) >= 2:
        print("\nMulti-temporal urban sprawl summary")
        print("-" * 70)
        print(f"{'Year':<6} | {'Urban km^2':<11} | {'Urban %':<8} | {'Nature loss km^2':<17} | {'Fractal D':<10}")
        print("-" * 70)

        base_yr = min(results.keys())
        base_forest = results[base_yr]["forest_km2"]

        for yr in sorted(results.keys()):
            res = results[yr]
            nature_loss = base_forest - res["forest_km2"] if yr != base_yr else 0.0
            print(
                f"{yr:<6} | {res['urban_km2']:<11.2f} | {res['urban_pct']:<8.2f} | "
                f"{nature_loss:<17.2f} | {res['fractal_dimension']:<10.4f}"
            )
        print("-" * 70)
