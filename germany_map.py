from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import numpy as np
from rasterio.mask import mask
from rasterio.plot import show

from matplotlib import cm
from matplotlib.colors import LightSource

def load_germany_geopandas() -> gpd.GeoDataFrame:
    def base():
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        world = gpd.read_file(url)
        germany_gdf = world[world["NAME_EN"] == "Germany"]

        if germany_gdf.empty:
            raise RuntimeError("Germany wurde im Datensatz nicht gefunden.")

        return germany_gdf
        # return germany.to_crs(epsg=3035)
    
    def add_height_profile(germany_gdf: gpd.GeoDataFrame):
        """
        Clipt ein DEM auf Deutschland und gibt Array + Metadaten zurück.
        """
        dem_path = "tifmap/eurodem.tif"
        with rasterio.open(dem_path) as src:
            
            germany_in_dem_crs = germany_gdf.to_crs(src.read_crs())

            out_image, out_transform = mask(
                src,
                germany_in_dem_crs.geometry,
                crop=True,
                nodata=src.nodata,
            )

            profile = src.profile.copy()
            profile.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
            )

        height = out_image[0].astype("float32")

        if profile.get("nodata") is not None:
            height[height == profile["nodata"]] = np.nan

        #height = (height - np.nanmin(height)) / (np.nanmax(height) - np.nanmin(height))
        return height, profile
    
    germany_base = base()
    
    print("add height map")
    germany2_height, height_profile = add_height_profile(germany_base)
    return germany2_height, height_profile
    

def render_map(germany: gpd.GeoDataFrame, output: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    germany.plot(
        ax=ax,
        color="#2c2c2c",
        edgecolor="#1f2937",
        linewidth=0.8,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
def render_height_map_2d(height: np.ndarray, output: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(height, cmap="terrain")

    ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_height_map_3d(height, profile, output: Path, dpi: int = 300) -> None:
    # NaN auffüllen, damit surface stabil rendert
    z = np.array(height, dtype=float)
    if np.isnan(z).any():
        z = np.where(np.isnan(z), np.nanmin(z), z)

    max_side = 900  # 500-1000 ist meist gut
    rows, cols = z.shape
    print(z.shape)
    step = max(1, int(np.ceil(max(rows, cols) / max_side)))
    z = z[::step, ::step]
    
    z_shade = z
    zmin, zmax = z.min(), z.max()
    if zmax > zmin:
        z = (z - zmin) / (zmax - zmin)

    rows, cols = z.shape
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    dx = abs(profile["transform"].a) * step
    dy = abs(profile["transform"].e) * step

    ls = LightSource(azdeg=100, altdeg=25)
    hillshade = ls.hillshade(z_shade, vert_exag=1.0, dx=dx, dy=dy)  # vert_exag höher testen: 10..40

    #hillshade = np.clip((hillshade - 0.35) * 2.4, 0, 1)
    surf = ax.plot_surface(
        X, Y, z,
        facecolors=cm.gray(hillshade),
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_axis_off()
    ax.set_zlim(0, 1)
    ax.invert_yaxis()

    ax.view_init(elev=72, azim=-90)
    ax.set_box_aspect((1, 1, 0.25))  # Höhe im Verhältnis flacher -> sieht oft besser aus

    fig.colorbar(surf, shrink=0.6, pad=0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
def create_files_for_blender(height: np.ndarray, profile: dict):
    output_dir = Path("output") / "blender"
    output_dir.mkdir(parents=True, exist_ok=True)

    z = np.array(height, dtype="float32")
    valid_mask = np.isfinite(z)
    if not np.any(valid_mask):
        raise RuntimeError("Keine gueltigen Hoehenwerte fuer Blender-Export.")

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))

    z_filled = z.copy()
    z_filled[~valid_mask] = z_min

    if z_max > z_min:
        z_norm = (z_filled - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z_filled, dtype="float32")

    height_u16 = np.clip(np.round(z_norm * 65535.0), 0, 65535).astype("uint16")

    height_png = output_dir / "height_16.png"
    with rasterio.open(
        height_png,
        "w",
        driver="PNG",
        width=height_u16.shape[1],
        height=height_u16.shape[0],
        count=1,
        dtype="uint16",
    ) as dst:
        dst.write(height_u16, 1)

    metadata = {
        "height_16_png": str(height_png),
        "width": int(z.shape[1]),
        "height": int(z.shape[0]),
        "z_min": z_min,
        "z_max": z_max,
        "pixel_size_x": float(abs(profile["transform"].a)),
        "pixel_size_y": float(abs(profile["transform"].e)),
        "crs": str(profile.get("crs")),
        "recommended_displacement_scale": float(z_max - z_min),
    }

    meta_path = output_dir / "terrain_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Blender Dateien gespeichert: {output_dir.resolve()}")
    
    
def load_germany_population():
    def base():
        
        print("load geopandas")
        url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
        world = gpd.read_file(url)
        germany_gdf = world[world["NAME_EN"] == "Germany"]

        if germany_gdf.empty:
            raise RuntimeError("Germany wurde im Datensatz nicht gefunden.")

        print("done")
        return germany_gdf
    
    import pandas as pd
    from shapely.geometry import box

    print("load pop")
    df = pd.read_csv("./tifmap/Zensus2022-1km-Gitter.csv", sep=";")
    df["Einwohner"] = pd.to_numeric(df["Einwohner"], errors="coerce").fillna(0)

    # 1km-Grid Mittelpunkte -> Punkte (meist EPSG:3035)
    pop_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_mp_1km"], df["y_mp_1km"]),
        crs="EPSG:3035",
    )

    # Punkte -> 1km Zellen
    half = 500
    pop_grid = pop_pts.copy()
    pop_grid["geometry"] = pop_grid.geometry.apply(
        lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half)
    )

    
    print("done")
    germany_gdf = base().to_crs(pop_grid.crs)
    pop_de = gpd.overlay(pop_grid, germany_gdf[["geometry"]], how="intersection")

    return pop_de


def create_population_heatmap_for_blender(pop_de: gpd.GeoDataFrame):
    output_dir = Path("output") / "blender"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_col = "x_mp_1km"
    y_col = "y_mp_1km"
    v_col = "Einwohner"

    grid = pop_de[[x_col, y_col, v_col]].copy()
    grid[v_col] = np.log1p(pd.to_numeric(grid[v_col], errors="coerce").fillna(0.0))
    grid = grid.groupby([x_col, y_col], as_index=False)[v_col].max()

    xs = np.sort(grid[x_col].unique())
    ys = np.sort(grid[y_col].unique())[::-1]

    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}

    heat = np.full((len(ys), len(xs)), np.nan, dtype="float32")
    for row in grid.itertuples(index=False):
        heat[y_index[getattr(row, y_col)], x_index[getattr(row, x_col)]] = getattr(row, v_col)

    valid = np.isfinite(heat)
    if not np.any(valid):
        raise RuntimeError("Keine gueltigen Populationswerte gefunden.")

    vmin = float(np.nanmin(heat))
    vmax = float(np.nanmax(heat))

    heat[~valid] = vmin
    if vmax > vmin:
        heat_norm = (heat - vmin) / (vmax - vmin)
    else:
        heat_norm = np.zeros_like(heat, dtype="float32")

    heat_u16 = np.clip(np.round(heat_norm * 65535.0), 0, 65535).astype("uint16")

    out_png = output_dir / "population_heat_16.png"
    with rasterio.open(
        out_png,
        "w",
        driver="PNG",
        width=heat_u16.shape[1],
        height=heat_u16.shape[0],
        count=1,
        dtype="uint16",
    ) as dst:
        dst.write(heat_u16, 1)

    # 1km -> 4km (4x4 Zellen zusammenfassen)
    agg = 4
    h, w = heat.shape
    h4 = h // agg
    w4 = w // agg
    heat_crop = heat[: h4 * agg, : w4 * agg]
    heat_4km = np.nansum(heat_crop.reshape(h4, agg, w4, agg), axis=(1, 3))

    vmin4 = float(np.nanmin(heat_4km))
    vmax4 = float(np.nanmax(heat_4km))
    if vmax4 > vmin4:
        heat_4km_norm = (heat_4km - vmin4) / (vmax4 - vmin4)
    else:
        heat_4km_norm = np.zeros_like(heat_4km, dtype="float32")

    heat_4km_u16 = np.clip(np.round(heat_4km_norm * 65535.0), 0, 65535).astype("uint16")

    out_png_4km = output_dir / "population_heat_4km_16.png"
    with rasterio.open(
        out_png_4km,
        "w",
        driver="PNG",
        width=heat_4km_u16.shape[1],
        height=heat_4km_u16.shape[0],
        count=1,
        dtype="uint16",
    ) as dst:
        dst.write(heat_4km_u16, 1)

    print(f"Population-Heatmap (1km) gespeichert: {out_png.resolve()}")
    print(f"Population-Heatmap (4km) gespeichert: {out_png_4km.resolve()}")


import pandas as pd
population_de = load_germany_population()
create_population_heatmap_for_blender(population_de)
#germany_height, profile = load_germany_geopandas()
#output_path = Path("output") / "germany_map.png"

#render_height_map_2d(germany, output_path, dpi=300)
#render_height_map_3d(germany_height, profile, Path("output") / "germany_height_3d.png", dpi=320)
#print(f"Karte gespeichert: {output_path}")
#create_files_for_blender(germany_height, profile)
