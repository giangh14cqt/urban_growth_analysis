import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Import our morphology and descriptors modules
from src.morphology import clean_urban_mask, clean_vegetation_mask, clean_water_mask
from src.descriptors import analyze_urban_features

def run_multi_spectral_segmentation(year, data_dir="data"):
    """
    Stacks spectral bands and advanced image features, runs K-Means clustering,
    automatically labels classes, applies morphology, and generates sprawl statistics.
    """
    print(f"\n=======================================================")
    print(f"--- Running Multi-Spectral Segmentation for Year {year} ---")
    print(f"=======================================================")
    
    # 1. Load our aligned bands and intermediate feature maps
    aligned = np.load(os.path.join(data_dir, f"aligned_{year}.npy"))
    ndvi = np.load(os.path.join(data_dir, f"ndvi_{year}.npy"))
    fourier = np.load(os.path.join(data_dir, f"fourier_{year}.npy"))
    saturation = np.load(os.path.join(data_dir, f"saturation_{year}.npy"))
    
    H, W, C = aligned.shape
    
    # Extract R, G, B, NIR bands (indices 0, 1, 2, 3)
    red = aligned[:, :, 0]
    green = aligned[:, :, 1]
    blue = aligned[:, :, 2]
    nir = aligned[:, :, 3]
    
    # 2. Reshape features to stack into N x 7 pixel array
    # Features: Red, Green, Blue, NIR, NDVI, Fourier, Saturation
    features = np.stack([
        red.ravel(),
        green.ravel(),
        blue.ravel(),
        nir.ravel(),
        ndvi.ravel(),
        fourier.ravel(),
        saturation.ravel()
    ], axis=1)
    
    # Handle any potential NaNs or infs just in case
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 3. Fit K-Means
    print("Fitting unsupervised K-Means model on 1,000,000 pixel stack...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Reshape labels back to 2D image coordinates
    raw_segmentation = labels.reshape((H, W))
    
    # 4. Physical-Property Cluster Mapping (Deterministic Labeling)
    centers = kmeans.cluster_centers_  # shape (3, 7)
    
    # Forest cluster has the absolute highest NDVI center (index 4)
    forest_cluster_idx = np.argmax(centers[:, 4])
    
    # Water cluster has the lowest NIR reflectance (index 3) and low NDVI (index 4)
    remaining_clusters = [i for i in range(3) if i != forest_cluster_idx]
    water_cluster_idx = remaining_clusters[np.argmin(centers[remaining_clusters, 3])]
    
    # Urban is the final remaining cluster
    urban_cluster_idx = [i for i in range(3) if i not in [forest_cluster_idx, water_cluster_idx]][0]
    
    print(f"Auto-mapped Clusters -> Forest: Cluster {forest_cluster_idx} | Water: Cluster {water_cluster_idx} | Urban: Cluster {urban_cluster_idx}")
    
    # 5. Extract raw binary masks based on K-Means clusters
    raw_urban = (raw_segmentation == urban_cluster_idx)
    raw_forest = (raw_segmentation == forest_cluster_idx)
    raw_water = (raw_segmentation == water_cluster_idx)
    
    # 6. Apply Phase 5 Morphological Cleaning to eliminate shadows, cars, and speckles
    print("Applying morphological opening/closing filters to refine spatial borders...")
    cleaned_urban = clean_urban_mask(raw_urban)
    cleaned_forest = clean_vegetation_mask(raw_forest)
    cleaned_water = clean_water_mask(raw_water)
    
    # Enforce priority to resolve overlap (since morphology might overlap boundaries slightly)
    # Order: Water has highest physical priority, then Forest, then Urban
    final_segmentation = np.zeros((H, W), dtype=np.uint8)
    final_segmentation[cleaned_urban > 0] = 1   # Class 1: Urban (Concrete)
    final_segmentation[cleaned_forest > 0] = 2  # Class 2: Forest/Grass (Nature)
    final_segmentation[cleaned_water > 0] = 3   # Class 3: Water (Rivers/Lakes)
    
    # Save the final classified numpy map
    np.save(os.path.join(data_dir, f"classified_{year}.npy"), final_segmentation)
    print(f"Saved final classified array to classified_{year}.npy")
    
    # 7. Compute Land Cover Sprawl Statistics
    total_pixels = H * W
    urban_pixels = np.sum(final_segmentation == 1)
    forest_pixels = np.sum(final_segmentation == 2)
    water_pixels = np.sum(final_segmentation == 3)
    unclassified_pixels = total_pixels - (urban_pixels + forest_pixels + water_pixels)
    
    # Pixel to area conversion: 1 pixel = 10m x 10m = 100 sq meters = 0.0001 sq km
    px_to_sq_km = 0.0001
    
    urban_area_km2 = urban_pixels * px_to_sq_km
    forest_area_km2 = forest_pixels * px_to_sq_km
    water_area_km2 = water_pixels * px_to_sq_km
    
    urban_pct = (urban_pixels / total_pixels) * 100
    forest_pct = (forest_pixels / total_pixels) * 100
    water_pct = (water_pixels / total_pixels) * 100
    
    print("\n--- Land Cover Statistics ---")
    print(f"Urban Area: {urban_area_km2:.2f} sq km ({urban_pct:.2f}%)")
    print(f"Forest/Vegetation Area: {forest_area_km2:.2f} sq km ({forest_pct:.2f}%)")
    print(f"Water Coverage: {water_area_km2:.2f} sq km ({water_pct:.2f}%)")
    
    # 8. Run Phase 5 Connected Components & Fractal Descriptors on final cleaned Urban mask
    descriptors = analyze_urban_features(aligned, cleaned_urban, year, data_dir)
    
    # 9. Save beautiful, publication-grade classified visual map
    # Color palette: Urban = Dark Grey, Forest = Green, Water = Blue, Background = Light Tan
    cmap_custom = plt.matplotlib.colors.ListedColormap(['#dcdcdc', '#555555', '#2ca02c', '#1f77b4'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # True Color Reference
    rgb = aligned[:, :, :3]
    rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
    axes[0].imshow(rgb_enhanced)
    axes[0].set_title(f"Native True Color Image - {year}")
    axes[0].axis("off")
    
    # Classified Map
    im = axes[1].imshow(final_segmentation, cmap=cmap_custom, vmin=0, vmax=3)
    axes[1].set_title(f"Unsupervised Multi-Spectral Land Cover Classification - {year}")
    axes[1].axis("off")
    
    # Add a custom legend
    colors = ['#555555', '#2ca02c', '#1f77b4']
    labels_legend = ["Urban (Concrete/Roads)", "Forest & Grass (Nature)", "Water (Vistula River)"]
    patches = [plt.matplotlib.patches.Patch(color=colors[i], label=labels_legend[i]) for i in range(len(colors))]
    plt.legend(handles=patches, loc="lower right", borderaxespad=1.0)
    
    png_path = os.path.join(data_dir, f"segmentation_report_{year}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved land cover classification visual report to {png_path}")
    
    return {
        "year": year,
        "urban_km2": urban_area_km2,
        "urban_pct": urban_pct,
        "forest_km2": forest_area_km2,
        "forest_pct": forest_pct,
        "water_km2": water_area_km2,
        "water_pct": water_pct,
        "fractal_dimension": descriptors["fractal_dimension"],
        "num_urban_patches": descriptors["num_patches"]
    }

if __name__ == "__main__":
    # Test script standalone
    data_dir = "data"
    years = [2018, 2022, 2025]
    
    results = {}
    for yr in years:
        if os.path.exists(os.path.join(data_dir, f"aligned_{yr}.npy")):
            res = run_multi_spectral_segmentation(yr, data_dir)
            results[yr] = res
        else:
            print(f"Skip {yr}: aligned_{yr}.npy not found.")
            
    # Print Multi-temporal Summary table
    if len(results) >= 2:
        print("\n" + "="*50)
        print("   MULTI-TEMPORAL URBAN SPRAWL SUMMARY (2018 - 2025)")
        print("="*50)
        print(f"{'Year':<6} | {'Urban Area (km2)':<16} | {'Urban %':<8} | {'Nature Loss (km2)':<17} | {'Fractal Dim':<11}")
        print("-"*65)
        
        base_yr = min(results.keys())
        base_forest = results[base_yr]["forest_km2"]
        
        for yr in sorted(results.keys()):
            res = results[yr]
            nature_loss_from_base = base_forest - res["forest_km2"] if yr != base_yr else 0.0
            print(f"{yr:<6} | {res['urban_km2']:<16.2f} | {res['urban_pct']:<8.2f} | {nature_loss_from_base:<17.2f} | {res['fractal_dimension']:<11.4f}")
        print("="*65)
