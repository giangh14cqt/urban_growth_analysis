import os
import math

import numpy as np
from dotenv import load_dotenv
from PIL import Image
from sentinelhub import BBox, CRS, DataCollection, MimeType, SentinelHubRequest, SHConfig


def compute_bbox(lat, lon, size_km):
    """
    Build a bounding box of `size_km` x `size_km` centered at (lat, lon).

    One degree of latitude is about 111.32 km everywhere. One degree of
    longitude shrinks with latitude, so we scale it by cos(lat).
    """
    delta_lat = size_km / 111.32
    delta_lon = size_km / (111.32 * math.cos(math.radians(lat)))

    min_lat = lat - delta_lat / 2
    max_lat = lat + delta_lat / 2
    min_lon = lon - delta_lon / 2
    max_lon = lon + delta_lon / 2

    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


def fetch_multi_spectral_sentinel(lat, lon, year, size_km=10.0, pixel_size=1000, output_dir="data"):
    """
    Download Sentinel-2 L2A bands (Red, Green, Blue, NIR + data mask) for the
    given location and year. Save the raw float32 array and a quick RGB preview.
    """
    load_dotenv()
    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")

    if not client_id or not client_secret or "your_sentinel_hub" in client_id:
        raise ValueError(
            "Sentinel Hub credentials not found in .env. "
            "Please set SH_CLIENT_ID and SH_CLIENT_SECRET."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    # Ask the API for 5 bands as float32 reflectance, not 8-bit visuals.
    # Keeping the full precision is important for NDVI and clustering later.
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
    print(f"\n--- Fetching data for year {year} ---")
    print(f"BBox: {bbox.min_x:.4f}, {bbox.min_y:.4f} to {bbox.max_x:.4f}, {bbox.max_y:.4f}")

    # Summer window: trees are in full leaf and there is no snow,
    # so NDVI and class colors stay comparable across years.
    time_interval = (f"{year}-06-01", f"{year}-08-31")

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,  # bottom-of-atmosphere reflectance
                time_interval=time_interval,
                maxcc=0.05,  # at most 5% cloud cover
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(pixel_size, pixel_size),
        config=config,
    )

    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"raw_{year}.npy")
    png_path = os.path.join(output_dir, f"visual_{year}.png")

    print("Sending request to Copernicus Sentinel Hub...")
    try:
        response_data = request.get_data()
        if len(response_data) == 0:
            print(f"No clear image found for year {year} in the summer window.")
            return None

        img_array = response_data[-1]  # shape (H, W, 5)

        np.save(npy_path, img_array)
        print(f"Saved raw multi-spectral data to {npy_path}")

        # Build a quick RGB preview. The x2.5 gain stretches typical land
        # reflectance (~0-0.4) into the visible 0-1 range.
        rgb = img_array[:, :, :3]
        rgb_uint8 = (np.clip(rgb * 2.5, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(png_path)
        print(f"Saved RGB preview to {png_path}")

        return img_array
    except Exception as e:
        print(f"Failed to fetch data for year {year}: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 L2A RGB+NIR imagery.")
    parser.add_argument("--lat", type=float, default=52.2297, help="Center latitude (default: Warsaw)")
    parser.add_argument("--lon", type=float, default=21.0122, help="Center longitude (default: Warsaw)")
    parser.add_argument("--year", type=int, required=True, help="Target year")
    parser.add_argument("--distance", type=float, default=10.0, help="Bounding box side in km (default: 10)")

    args = parser.parse_args()

    # 10 m / pixel -> distance_km * 100 pixels per side.
    pixel_dim = int(args.distance * 100)

    print(f"Target area: {args.distance} km x {args.distance} km at ({args.lat}, {args.lon})")
    print(f"Output grid: {pixel_dim} x {pixel_dim} at 10 m/pixel")

    fetch_multi_spectral_sentinel(
        lat=args.lat,
        lon=args.lon,
        year=args.year,
        size_km=args.distance,
        pixel_size=pixel_dim,
    )
