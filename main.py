import os
import argparse
import numpy as np

# Import all core modular pipelines
from src.sentinel_api import fetch_multi_spectral_sentinel
from src.alignment import register_and_align
from src.enhancement import process_enhancements
from src.fourier_analysis import process_fourier_analysis
from src.segmentation import run_multi_spectral_segmentation

def run_pipeline(dataset, lat=None, lon=None, distance_km=None, years=None, data_dir="data"):
    """
    Coordinates the end-to-end multi-temporal satellite pipeline:
    1. Loads pre-cached Warsaw datasets (offline) or downloads from Sentinel-2 L2A API.
    2. Registers and aligns all target years perfectly to the oldest base year.
    3. Calculates NDVI, saturation indices, and applies LAB CLAHE visual contrast.
    4. Computes sliding-window 2D FFT texture maps to trace repeating road and block grids.
    5. Performs unsupervised K-Means land cover classification and cleans masks via morphology.
    6. Summarizes multi-temporal statistics, nature loss curves, and boundary fractal complexity.
    """
    if dataset == "warsaw":
        # Force default Warsaw parameters and run strictly offline
        lat = 52.2297
        lon = 21.0122
        distance_km = 10.0
        years = [2018, 2022, 2025]
        print("\n" + "="*70)
        print("      MULTI-TEMPORAL URBAN SPRAWL PIPELINE CENTRAL ORCHESTRATOR")
        print("="*70)
        print(f"Dataset Option     : Default Warsaw (OFFLINE MODE)")
        print(f"Center Coordinates : ({lat:.6f}, {lon:.6f})")
        print(f"Study Grid Bounds  : {distance_km:.2f} km x {distance_km:.2f} km")
        print(f"Target Years       : {years}")
        print(f"Output Directory   : '{data_dir}'")
        print("="*70)
    else:
        # Custom dataset mode (fetches using Sentinel Hub API)
        print("\n" + "="*70)
        print("      MULTI-TEMPORAL URBAN SPRAWL PIPELINE CENTRAL ORCHESTRATOR")
        print("="*70)
        print(f"Dataset Option     : Custom API Dataset (ONLINE MODE)")
        print(f"Center Coordinates : ({lat:.6f}, {lon:.6f})")
        print(f"Study Grid Bounds  : {distance_km:.2f} km x {distance_km:.2f} km")
        print(f"Target Years       : {years}")
        print(f"Output Directory   : '{data_dir}'")
        print("="*70)
    
    os.makedirs(data_dir, exist_ok=True)
    pixel_dim = int(distance_km * 100) # 10m spatial resolution
    
    # --- PHASE 2: MULTI-SPECTRAL DATA ACQUISITION ---
    print("\n>>> PHASE 2: Multi-Spectral Data Acquisition")
    for yr in years:
        raw_path = os.path.join(data_dir, f"raw_{yr}.npy")
        
        if dataset == "warsaw":
            preset_raw_path = os.path.join("presets", f"raw_{yr}.npy")
            if not os.path.exists(preset_raw_path):
                raise FileNotFoundError(
                    f"\n[Offline Mode Error]: Pre-cached Warsaw raw file not found at '{preset_raw_path}'.\n"
                    "Please ensure the default Warsaw raw data files are present in the 'presets/' folder."
                )
            
            if not os.path.exists(raw_path):
                import shutil
                print(f"✓ Copying pre-cached Warsaw raw data for {yr} from 'presets/' to '{data_dir}'...")
                shutil.copy2(preset_raw_path, raw_path)
            else:
                print(f"✓ Raw Warsaw multi-spectral data for {yr} already exists in '{raw_path}'.")
        else:
            # Custom dataset mode
            if os.path.exists(raw_path):
                print(f"✓ Raw multi-spectral data for {yr} already exists in '{raw_path}'.")
            else:
                # Fetch using Sentinel Hub API
                from dotenv import load_dotenv
                load_dotenv()
                client_id = os.getenv("SH_CLIENT_ID")
                client_secret = os.getenv("SH_CLIENT_SECRET")
                if not client_id or not client_secret or "your_sentinel_hub" in client_id:
                    raise ValueError(
                        f"\n[API Credentials Error]: Sentinel Hub credentials not found in your .env file.\n"
                        "To download data for custom study areas, please configure SH_CLIENT_ID and SH_CLIENT_SECRET in .env."
                    )
                
                fetch_multi_spectral_sentinel(
                    lat=lat,
                    lon=lon,
                    year=yr,
                    size_km=distance_km,
                    pixel_size=pixel_dim,
                    output_dir=data_dir
                )
            
    # Check that all raw files exist before proceeding
    for yr in years:
        raw_path = os.path.join(data_dir, f"raw_{yr}.npy")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Error: Missing raw satellite data for year {yr}. Cannot proceed with alignment.")
            
    # --- PHASE 3: IMAGE REGISTRATION & ALIGNMENT ---
    print("\n>>> PHASE 3: Multi-Spectral Keypoint Registration")
    base_yr = min(years)
    base_raw_path = os.path.join(data_dir, f"raw_{base_yr}.npy")
    base_raw = np.load(base_raw_path)
    
    # Save the base reference year as aligned directly to ensure uniform pipeline structures
    np.save(os.path.join(data_dir, f"aligned_{base_yr}.npy"), base_raw)
    print(f"✓ Baseline reference frame set to oldest year: {base_yr}. Saved aligned_{base_yr}.npy")
    
    for yr in years:
        if yr == base_yr:
            continue
        target_raw = np.load(os.path.join(data_dir, f"raw_{yr}.npy"))
        register_and_align(base_raw, target_raw, yr, data_dir)
        
    # --- PHASE 4: SPECTRAL ENHANCEMENTS & FFT TEXTURES ---
    print("\n>>> PHASE 4: Feature Engineering (NDVI, Saturation, CLAHE, FFT)")
    for yr in years:
        process_enhancements(yr, data_dir)
        process_fourier_analysis(yr, data_dir)
        
    # --- PHASE 6: SEGMENTATION & CLASSIFICATION ---
    print("\n>>> PHASE 6: Unsupervised Pixel Clustering & Spatial Analysis")
    results = {}
    for yr in years:
        res = run_multi_spectral_segmentation(yr, data_dir)
        results[yr] = res
        
    # --- MULTI-TEMPORAL SCIENTIFIC REPORT ---
    if len(results) >= 2:
        print("\n" + "="*75)
        print("                 MULTI-TEMPORAL URBAN SPRAWL SUMMARY REPORT")
        print("="*75)
        print(f"{'Year':<6} | {'Urban Area (km2)':<16} | {'Urban %':<8} | {'Nature Loss (km2)':<17} | {'Fractal Dim':<11}")
        print("-"*75)
        
        base_forest = results[base_yr]["forest_km2"]
        
        for yr in sorted(results.keys()):
            res = results[yr]
            nature_loss_from_base = base_forest - res["forest_km2"] if yr != base_yr else 0.0
            print(f"{yr:<6} | {res['urban_km2']:<16.2f} | {res['urban_pct']:<8.2f} | {nature_loss_from_base:<17.2f} | {res['fractal_dimension']:<11.4f}")
        print("="*75)
    else:
        print("\nWarning: Need at least 2 target years to compile a multi-temporal report.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Orchestrator for Multi-Temporal Urban Growth.")
    parser.add_argument("--dataset", type=str, choices=["warsaw", "custom"], default="warsaw",
                        help="Select dataset option: 'warsaw' (default, uses pre-cached offline data) or 'custom' (requires Sentinel Hub API keys to download data)")
    parser.add_argument("--lat", type=float, help="Latitude of the center point (only used for '--dataset custom')")
    parser.add_argument("--lon", type=float, help="Longitude of the center point (only used for '--dataset custom')")
    parser.add_argument("--distance", type=float, help="Width/height of study area in km (only used for '--dataset custom')")
    parser.add_argument("--years", type=str, help="Comma-separated target years, e.g. 2018,2022,2025 (only used for '--dataset custom')")
    parser.add_argument("--data_dir", type=str, default="data", help="Output directory (default: data)")
    
    args = parser.parse_args()
    
    if args.dataset == "warsaw":
        # Ignore custom params and run standard warsaw offline pipeline
        run_pipeline(
            dataset="warsaw",
            data_dir=args.data_dir
        )
    else:
        # Check custom inputs
        if args.lat is None or args.lon is None or args.distance is None or args.years is None:
            raise ValueError(
                "Error: When using '--dataset custom', you must specify all parameters: --lat, --lon, --distance, and --years."
            )
            
        # Parse comma-separated list of years into integers
        target_years = [int(yr.strip()) for yr in args.years.split(",") if yr.strip()]
        target_years = sorted(list(set(target_years)))  # Ensure unique, sorted years
        
        if len(target_years) == 0:
            raise ValueError("Error: You must provide at least one target year.")
            
        run_pipeline(
            dataset="custom",
            lat=args.lat,
            lon=args.lon,
            distance_km=args.distance,
            years=target_years,
            data_dir=args.data_dir
        )
