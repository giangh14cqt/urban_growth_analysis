import os
import math
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
from PIL import Image
import numpy as np

def compute_bbox(lat, lon, size_km):
    """
    Computes a Bounding Box (min_lon, min_lat, max_lon, max_lat) 
    centered at (lat, lon) with a width and height of size_km.
    """
    # 1 degree of latitude is roughly 111.32 km
    delta_lat = size_km / 111.32
    
    # 1 degree of longitude is roughly 111.32 * cos(lat) km
    delta_lon = size_km / (111.32 * math.cos(math.radians(lat)))
    
    min_lat = lat - delta_lat / 2
    max_lat = lat + delta_lat / 2
    min_lon = lon - delta_lon / 2
    max_lon = lon + delta_lon / 2
    
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

def fetch_multi_spectral_sentinel(lat, lon, year, size_km=10.0, pixel_size=1000, output_dir="data"):
    """
    Fetches raw Float32 multi-spectral bands (Red, Green, Blue, NIR) from Sentinel Hub.
    Saving raw data as .npy for downstream analysis and generating a visual PNG.
    """
    load_dotenv()
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")
    
    if not client_id or not client_secret or "your_sentinel_hub" in client_id:
        raise ValueError("Sentinel Hub credentials not found in .env. Please configure SH_CLIENT_ID and SH_CLIENT_SECRET.")
        
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    
    # Evalscript to return FLOAT32 reflectance bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
    # plus the dataMask to filter valid pixels.
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02", "B03", "B04", "B08", "dataMask"],
        output: { bands: 5, sampleType: "FLOAT32" }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02, sample.B08, sample.dataMask];
    }
    """
    
    bbox = compute_bbox(lat, lon, size_km)
    print(f"\n--- Fetching Data for Year {year} ---")
    print(f"BBox: {bbox.min_x:.4f}, {bbox.min_y:.4f} to {bbox.max_x:.4f}, {bbox.max_y:.4f}")
    
    # We query the summer season of the target year to avoid snow cover and get optimal foliage
    time_interval = (f"{year}-06-01", f"{year}-08-31")
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,  # L2A provides bottom-of-atmosphere surface reflectance
                time_interval=time_interval,
                maxcc=0.05  # Max 5% cloud cover
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=(pixel_size, pixel_size),
        config=config
    )
    
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"raw_{year}.npy")
    png_path = os.path.join(output_dir, f"visual_{year}.png")
    
    print("Executing request to Copernicus Sentinel Hub...")
    try:
        response_data = request.get_data()
        if len(response_data) > 0:
            # Shape is (H, W, 5)
            img_array = response_data[-1]
            
            # Save raw FLOAT32 array for scientific accuracy
            np.save(npy_path, img_array)
            print(f"Saved raw multi-spectral data to {npy_path}")
            
            # Extract RGB bands for visual display (indices 0: Red, 1: Green, 2: Blue)
            rgb = img_array[:, :, :3]
            
            # Apply standard gain factor of 2.5 for visual contrast, clip, and scale to 0-255
            rgb_enhanced = np.clip(rgb * 2.5, 0.0, 1.0)
            rgb_uint8 = (rgb_enhanced * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(rgb_uint8)
            img_pil.save(png_path)
            print(f"Saved visual RGB image to {png_path}")
            
            return img_array
        else:
            print(f"No clear image found for year {year} during the summer interval.")
            return None
    except Exception as e:
        print(f"Failed to fetch data for year {year}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch multi-spectral (RGB+NIR) Sentinel-2 L2A images.")
    parser.add_argument("--lat", type=float, default=52.2297, help="Latitude of the center point (default: Warsaw)")
    parser.add_argument("--lon", type=float, default=21.0122, help="Longitude of the center point (default: Warsaw)")
    parser.add_argument("--year", type=int, required=True, help="Target year to fetch")
    parser.add_argument("--distance", type=float, default=10.0, help="Bounding box size in km (default: 10.0)")
    
    args = parser.parse_args()
    
    # 10m spatial resolution means: (distance * 1000m) / 10m_per_pixel = distance * 100
    pixel_dim = int(args.distance * 100)
    
    print(f"Target Area: {args.distance}km x {args.distance}km centered at ({args.lat}, {args.lon})")
    print(f"Resolution Enforced: 10m/pixel (Output grid: {pixel_dim}x{pixel_dim})")
    
    fetch_multi_spectral_sentinel(
        lat=args.lat,
        lon=args.lon,
        year=args.year,
        size_km=args.distance,
        pixel_size=pixel_dim
    )
