import argparse
import os

import numpy as np

from src.alignment import register_and_align
from src.enhancement import process_enhancements
from src.fourier_analysis import process_fourier_analysis
from src.segmentation import run_multi_spectral_segmentation
from src.sentinel_api import fetch_multi_spectral_sentinel


def run_pipeline(dataset, lat=None, lon=None, distance_km=None, years=None, data_dir="data"):
    """
    Run the full pipeline end to end:
      1. Load pre-cached Warsaw arrays (offline) or fetch from Sentinel Hub (online).
      2. Align every target year to the oldest one.
      3. Compute NDVI, saturation, CLAHE and FFT texture maps.
      4. Cluster pixels with K-Means and clean masks with morphology.
      5. Print area statistics, nature loss and boundary fractal dimension.
    """
    if dataset == "warsaw":
        lat = 52.2297
        lon = 21.0122
        distance_km = 10.0
        years = [2018, 2022, 2025]
        mode_label = "Warsaw preset (offline)"
    else:
        mode_label = "Custom area (Sentinel Hub)"

    print("\nMulti-temporal urban sprawl pipeline")
    print("-" * 60)
    print(f"Mode      : {mode_label}")
    print(f"Center    : ({lat:.6f}, {lon:.6f})")
    print(f"Area      : {distance_km:.2f} km x {distance_km:.2f} km")
    print(f"Years     : {years}")
    print(f"Output    : '{data_dir}'")
    print("-" * 60)

    os.makedirs(data_dir, exist_ok=True)
    pixel_dim = int(distance_km * 100)  # 10 m per pixel

    # --- Data acquisition ---
    print("\n[1/4] Data acquisition")
    for yr in years:
        raw_path = os.path.join(data_dir, f"raw_{yr}.npy")

        if dataset == "warsaw":
            preset_raw_path = os.path.join("presets", f"raw_{yr}.npy")
            if not os.path.exists(preset_raw_path):
                raise FileNotFoundError(
                    f"Preset raw file not found at '{preset_raw_path}'. "
                    "Make sure the Warsaw raw arrays are present in 'presets/'."
                )

            if not os.path.exists(raw_path):
                import shutil

                print(f"  Copying preset raw data for {yr} from 'presets/' to '{data_dir}'...")
                shutil.copy2(preset_raw_path, raw_path)
            else:
                print(f"  Raw data for {yr} already in '{raw_path}'.")
        else:
            if os.path.exists(raw_path):
                print(f"  Raw data for {yr} already in '{raw_path}'.")
            else:
                from dotenv import load_dotenv

                load_dotenv()
                client_id = os.getenv("SH_CLIENT_ID")
                client_secret = os.getenv("SH_CLIENT_SECRET")
                if not client_id or not client_secret or "your_sentinel_hub" in client_id:
                    raise ValueError(
                        "Sentinel Hub credentials not found in .env. "
                        "Set SH_CLIENT_ID and SH_CLIENT_SECRET to download custom areas."
                    )

                fetch_multi_spectral_sentinel(
                    lat=lat,
                    lon=lon,
                    year=yr,
                    size_km=distance_km,
                    pixel_size=pixel_dim,
                    output_dir=data_dir,
                )

    for yr in years:
        raw_path = os.path.join(data_dir, f"raw_{yr}.npy")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw data for year {yr}; cannot continue.")

    # --- Alignment ---
    print("\n[2/4] Alignment")
    base_yr = min(years)
    base_raw = np.load(os.path.join(data_dir, f"raw_{base_yr}.npy"))

    np.save(os.path.join(data_dir, f"aligned_{base_yr}.npy"), base_raw)
    print(f"  Reference year: {base_yr}. Saved aligned_{base_yr}.npy.")

    for yr in years:
        if yr == base_yr:
            continue
        target_raw = np.load(os.path.join(data_dir, f"raw_{yr}.npy"))
        register_and_align(base_raw, target_raw, yr, data_dir)

    # --- Feature engineering ---
    print("\n[3/4] Feature engineering (NDVI, saturation, CLAHE, FFT)")
    for yr in years:
        process_enhancements(yr, data_dir)
        process_fourier_analysis(yr, data_dir)

    # --- Segmentation and analysis ---
    print("\n[4/4] Segmentation and spatial analysis")
    results = {}
    for yr in years:
        results[yr] = run_multi_spectral_segmentation(yr, data_dir)

    if len(results) >= 2:
        print("\nMulti-temporal urban sprawl summary")
        print("-" * 75)
        print(f"{'Year':<6} | {'Urban km^2':<11} | {'Urban %':<8} | {'Nature loss km^2':<17} | {'Fractal D':<10}")
        print("-" * 75)

        base_forest = results[base_yr]["forest_km2"]
        for yr in sorted(results.keys()):
            res = results[yr]
            nature_loss = base_forest - res["forest_km2"] if yr != base_yr else 0.0
            print(
                f"{yr:<6} | {res['urban_km2']:<11.2f} | {res['urban_pct']:<8.2f} | "
                f"{nature_loss:<17.2f} | {res['fractal_dimension']:<10.4f}"
            )
        print("-" * 75)
    else:
        print("\nNeed at least 2 years to produce a multi-temporal report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-temporal urban growth pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["warsaw", "custom"],
        default="warsaw",
        help="'warsaw' uses the cached preset arrays; 'custom' downloads from Sentinel Hub.",
    )
    parser.add_argument("--lat", type=float, help="Center latitude (only for --dataset custom)")
    parser.add_argument("--lon", type=float, help="Center longitude (only for --dataset custom)")
    parser.add_argument("--distance", type=float, help="Area side in km (only for --dataset custom)")
    parser.add_argument("--years", type=str, help="Comma-separated years, e.g. 2018,2022,2025 (custom only)")
    parser.add_argument("--data_dir", type=str, default="data", help="Output directory (default: data)")

    args = parser.parse_args()

    if args.dataset == "warsaw":
        run_pipeline(dataset="warsaw", data_dir=args.data_dir)
    else:
        if args.lat is None or args.lon is None or args.distance is None or args.years is None:
            raise ValueError(
                "When using --dataset custom you must pass --lat, --lon, --distance and --years."
            )

        target_years = sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
        if not target_years:
            raise ValueError("Provide at least one target year.")

        run_pipeline(
            dataset="custom",
            lat=args.lat,
            lon=args.lon,
            distance_km=args.distance,
            years=target_years,
            data_dir=args.data_dir,
        )
