# Fire Event Toolkit

Python tools for turning satellite active-fire **hotspots** and mapped **burnt-area polygons** into:

1. a continuous **day-of-burning (DOY) raster**, and
2. individual **fire events** with size statistics.

This is a generic implementation of the **fire event delineation (FED)** method described in:

> Fisher, R., Mollick, P., Woolley, L.-A., Vigilante, T., Russ, L. & Wunambal Gaambera Aboriginal Corporation (2026). A method for characterising fire events in high fire frequency landscapes: case study application to North Kimberley fire regimes, Western Australia. *International Journal of Wildland Fire*, 35(10), WF26071. https://doi.org/10.1071/WF26071

The paper applies the method to 2014–2023 fire mapping of North Kimberley savannas. The code here has no regional settings. It works with any hotspot product (MODIS, VIIRS, Himawari and others) and any burnt-area / fire-scar map with a month attribute.

```
hotspots (points, DOY)  ──┐
                          ├─► hotspot_doy_interpolation.py ─► DOY raster ─► fire_event_identification.py ─► events + stats
burnt areas (polygons,  ──┘
  month attribute)
```

## Installation

GDAL-based packages install most reliably with conda:

```bash
conda create -n fire-events -c conda-forge python=3.11 geopandas rasterio pandas numpy
conda activate fire-events
```

Or with pip: `pip install -r requirements.txt`

`hotspot_doy_interpolation.py` also needs [SAGA GIS](https://saga-gis.sourceforge.io/) (tested with 9.x). Put `saga_cmd` on your `PATH`, set the `SAGA_CMD` environment variable, or pass `--saga-cmd`.

## 1. Hotspot DOY interpolation

```bash
python hotspot_doy_interpolation.py \
    --hotspots "data/HotSpot*.shp" \
    --burnt-areas data/burnt_areas_2026.shp \
    --out-dir output \
    --year 2026 \
    --bbox 112.92 -34.50 154.00 -9.33 \
    --cell-size 0.0026
```

**How it works:**

- For each month, it selects the hotspots whose DOY falls in that month. Each window starts a few days before the month (`--overlap-days`, default 3), so the surface stays continuous at month boundaries.
- It interpolates hotspot DOY to a grid with SAGA's Multilevel B-Spline (`grid_spline` tool 4).
- It clips each monthly grid to the burnt-area polygons mapped for that month (`--month-field`).
- It mosaics the 12 clipped grids into one LZW-compressed int32 GeoTIFF. Value 0 means unburnt or no data, and 1–366 is the day of burning.

**Inputs:**

| Input | Requirement |
|---|---|
| Hotspots | Point layer with a DOY field (`--doy-field`, default `DOY`), or a date field (`--date-field`) that DOY is computed from |
| Burnt areas | Polygon layer with a month field holding 1–12 (`--month-field`, default `month`) |

Both layers should share a CRS. The burnt areas are reprojected to the grid CRS if they differ. `--bbox` and `--cell-size` are in the hotspot CRS. If you leave out `--bbox`, the hotspot extent is used.

Run `python hotspot_doy_interpolation.py --help` for all options.

## 2. Fire event identification

```bash
python fire_event_identification.py output/DOY_2026_Nov25.tif --out-dir events
# or a whole folder of DOY rasters
python fire_event_identification.py rasters/ --out-dir events --tolerance 1 --min-area-ha 10
```

**How it works:** the script uses region growing on the DOY raster. From each unlabelled burnt pixel, an event grows into neighbouring burnt pixels whose DOY differs from the *current* pixel by at most `--tolerance` days. A fire spreading steadily over many days therefore stays one event. Adjacent burns separated by a jump in date are split into separate events.

**Outputs** (per input raster):

| File | Contents |
|---|---|
| `regions_<name>.tif` | Event ID raster (int32, 0 = unburnt) |
| `<name>_GT<min>ha.shp` | Events larger than `--min-area-ha`, largest first |
| `<name>_top<N>.shp` | The `--top-n` largest events |
| `<name>_area_stats.csv` | Totals, largest events and a log-scale size-class table (0–1 ha … >1,000,000 ha) |
| `<name>_merged.shp` | All events dissolved into one footprint |

Areas are calculated in hectares. If the raster uses geographic coordinates, polygons are reprojected to an equal-area CRS first (`--area-crs`, default `EPSG:6933`). For regional work, a local equal-area CRS such as `EPSG:3577` for Australia gives more accurate areas.

> **Performance:** region growing is pure Python, so large high-resolution rasters can take a while. Clip to your area of interest first if needed.

## Citation

If you use this software, please cite the paper above. You can also cite the software itself using [`CITATION.cff`](CITATION.cff) or the Zenodo DOI shown on the repository page.

## Author

Rohan Fisher. 

## License

MIT © 2026 Rohan Fisher. See [LICENSE](LICENSE).
