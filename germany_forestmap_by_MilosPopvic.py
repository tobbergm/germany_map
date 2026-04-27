from pathlib import Path
from urllib.parse import urlparse, unquote

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge
from tqdm import tqdm


WORKDIR = Path(".")
OUTPUT_DIR = WORKDIR / "output" / "blender"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY = "DEU"
CRS_LONGLAT = "EPSG:4326"

# Funktionierende Quelle (Stand 2026-04-27)
URLS = [
    "https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif?download=1",
]

AGGREGATE_FACTOR = 2
BLUR_RADIUS = 30
BLUR_SIGMA = 7.0
DETAIL_WEIGHT = 0.30
COLOR_GAMMA = 0.75


def get_country_borders(country_iso3: str = COUNTRY) -> gpd.GeoDataFrame:
    url = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_10M_2020_4326.geojson"
    countries = gpd.read_file(url)
    country = countries[countries["ISO3_CODE"] == country_iso3]
    if country.empty:
        raise RuntimeError(f"ISO3 nicht gefunden: {country_iso3}")
    return country.to_crs(CRS_LONGLAT)


def download_file(url: str, out_dir: Path) -> Path:
    filename = unquote(Path(urlparse(url).path).name)
    out_path = out_dir / filename
    if out_path.exists():
        return out_path

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(out_path, "wb") as file, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=filename,
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))

    return out_path


def crop_single_raster_to_country(raster_file: Path, country_gdf: gpd.GeoDataFrame):
    with rasterio.open(raster_file) as dataset:
        geometries = [geom for geom in country_gdf.geometry]
        cropped, transform = mask(
            dataset,
            geometries,
            crop=True,
            filled=False,
        )
        cropped_meta = dataset.meta.copy()
        cropped_meta.update(
            {
                "height": cropped.shape[1],
                "width": cropped.shape[2],
                "transform": transform,
            }
        )
    return cropped, cropped_meta


def mosaic_and_crop(raster_files, country_gdf):
    if len(raster_files) == 1:
        return crop_single_raster_to_country(raster_files[0], country_gdf)

    srcs = [rasterio.open(f) for f in raster_files]
    mosaic_array, mosaic_transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(
        {
            "height": mosaic_array.shape[1],
            "width": mosaic_array.shape[2],
            "transform": mosaic_transform,
            "crs": CRS_LONGLAT,
        }
    )
    for src in srcs:
        src.close()

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(mosaic_array)
            cropped, transform = mask(
                dataset,
                [geom for geom in country_gdf.geometry],
                crop=True,
                filled=False,
            )
            cropped_meta = dataset.meta.copy()
            cropped_meta.update(
                {
                    "height": cropped.shape[1],
                    "width": cropped.shape[2],
                    "transform": transform,
                }
            )
    return cropped, cropped_meta


def downsample_mean(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    h, w = array.shape
    h2 = h // factor
    w2 = w // factor
    arr = array[: h2 * factor, : w2 * factor]
    return arr.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def gaussian_blur2d(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    k = gaussian_kernel1d(radius, sigma)

    pad = radius
    # axis=1
    p = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(img, dtype=np.float32)
    for r in range(img.shape[0]):
        out[r, :] = np.convolve(p[r, :], k, mode="valid")

    # axis=0
    p2 = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    out2 = np.empty_like(out, dtype=np.float32)
    for c in range(out.shape[1]):
        out2[:, c] = np.convolve(p2[:, c], k, mode="valid")

    return out2


def build_forest_texture(forest_raw: np.ndarray) -> np.ndarray:
    # Erwartung: Tree Cover Fraction in Prozent [0..100]
    forest = forest_raw.astype(np.float32)
    forest = np.where(np.isfinite(forest), forest, np.nan)
    forest = np.clip(forest, 0.0, 100.0)

    valid = np.isfinite(forest)
    forest_filled = np.where(valid, forest, 0.0)

    blur = gaussian_blur2d(forest_filled, BLUR_RADIUS, BLUR_SIGMA)
    detail = forest_filled - blur
    mixed = blur + DETAIL_WEIGHT * detail

    # weiche Kante an NoData-Rand statt harter Sprung
    valid_soft = gaussian_blur2d(valid.astype(np.float32), BLUR_RADIUS, BLUR_SIGMA)
    valid_soft = np.clip(valid_soft, 0.0, 1.0)
    mixed *= valid_soft

    mixed = np.clip(mixed, 0.0, 100.0)
    return mixed


def save_texture_png_16(texture_0_100: np.ndarray, out_path: Path) -> None:
    norm = np.clip(texture_0_100 / 100.0, 0.0, 1.0)
    tex_u16 = np.round(norm * 65535.0).astype(np.uint16)

    with rasterio.open(
        out_path,
        "w",
        driver="PNG",
        width=tex_u16.shape[1],
        height=tex_u16.shape[0],
        count=1,
        dtype="uint16",
    ) as dst:
        dst.write(tex_u16, 1)


def colorize_forest_texture(texture_0_100: np.ndarray) -> np.ndarray:
    values = texture_0_100[np.isfinite(texture_0_100)]
    if values.size == 0:
        norm = np.zeros_like(texture_0_100, dtype=np.float32)
    else:
        lo = np.percentile(values, 2.0)
        hi = np.percentile(values, 98.0)
        if hi <= lo:
            norm = np.clip(texture_0_100 / 100.0, 0.0, 1.0)
        else:
            norm = np.clip((texture_0_100 - lo) / (hi - lo), 0.0, 1.0)

    # Hebt mittlere/hohe Unterschiede sichtbarer hervor
    norm = np.power(norm, COLOR_GAMMA)

    # Realistische Palette: offen/hell -> Mischwald -> dichter Wald
    stops = np.array([0.0, 0.15, 0.35, 0.6, 0.8, 1.0], dtype=np.float32)
    colors = np.array(
        [
            [170, 170, 140],  # trockene/offene Flaechen
            [178, 187, 126],  # lockere Vegetation
            [132, 158, 96],   # buschig
            [87, 128, 77],    # Mischwald
            [52, 95, 57],     # dichter Wald
            [24, 58, 34],     # sehr dichter Wald
        ],
        dtype=np.float32,
    )

    r = np.interp(norm, stops, colors[:, 0])
    g = np.interp(norm, stops, colors[:, 1])
    b = np.interp(norm, stops, colors[:, 2])

    rgb = np.stack([r, g, b], axis=0)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def save_texture_rgb_png(texture_0_100: np.ndarray, out_path: Path) -> None:
    rgb = colorize_forest_texture(texture_0_100)
    with rasterio.open(
        out_path,
        "w",
        driver="PNG",
        width=rgb.shape[2],
        height=rgb.shape[1],
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(rgb)


def main():
    country_borders = get_country_borders(COUNTRY)
    raster_files = [download_file(url, WORKDIR) for url in URLS]
    forest_cropped, _ = mosaic_and_crop(raster_files, country_borders)

    if np.ma.isMaskedArray(forest_cropped):
        forest = forest_cropped[0].astype(np.float32).filled(np.nan)
    else:
        forest = forest_cropped[0].astype(np.float32)

    forest_ds = downsample_mean(forest, AGGREGATE_FACTOR)
    texture = build_forest_texture(forest_ds)

    out_bw = OUTPUT_DIR / "forest_cover_texture_16.png"
    out_color = OUTPUT_DIR / "forest_cover_texture_color.png"

    save_texture_png_16(texture, out_bw)
    save_texture_rgb_png(texture, out_color)

    print(f"Fertig (BW): {out_bw.resolve()}")
    print(f"Fertig (Color): {out_color.resolve()}")


if __name__ == "__main__":
    main()
