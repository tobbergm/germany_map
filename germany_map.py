from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


def load_germany_geopandas() -> gpd.GeoDataFrame:
    url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    germany = world[world["NAME_EN"] == "Germany"]

    if germany.empty:
        raise RuntimeError("Germany wurde im Datensatz nicht gefunden.")

    # return germany.to_crs(epsg=4326)
    return germany.to_crs(epsg=3035)


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

germany = load_germany()
output_path = Path("output") / "germany_map.png"

render_map(germany, output_path, dpi=300)
print(f"Karte gespeichert: {output_path}")
