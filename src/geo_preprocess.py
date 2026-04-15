from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class GeoRaster:
    rgb: np.ndarray  # uint8 HxWx3
    transform: "Affine"
    crs: object
    nodata_mask: Optional[np.ndarray] = None  # bool HxW, True where nodata/invalid


@dataclass(frozen=True)
class GeoTile:
    case_name: str
    x0: int
    y0: int
    rgb: np.ndarray  # uint8 thxtwx3
    transform: "Affine"
    crs: object
    full_height: int
    full_width: int


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
        from affine import Affine  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "FBIS geospatial mode requires `rasterio` (and `affine`). "
            "Install via conda-forge (recommended) or pip."
        ) from e


def _percentile_stretch_to_uint8(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)

    vmin = np.percentile(x[finite], p_low)
    vmax = np.percentile(x[finite], p_high)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(x[finite]))
        vmax = float(np.max(x[finite]))
        if vmax <= vmin:
            return np.zeros_like(x, dtype=np.uint8)

    y = (x - vmin) / (vmax - vmin)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).round().astype(np.uint8)


def read_geotiff_as_rgb(tif_path: str) -> GeoRaster:
    """
    Read a (possibly multi-band) GeoTIFF and convert to an RGB uint8 image.
    Georeferencing is preserved via `transform` + `crs`.
    """
    _require_rasterio()
    import rasterio

    with rasterio.open(tif_path) as ds:
        # Read as (bands, H, W)
        arr = ds.read()
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata

    if arr.ndim != 3:
        raise ValueError(f"Unexpected raster shape for {tif_path}: {arr.shape}")

    bands, h, w = arr.shape

    # Build nodata mask from nodata value if present, else keep None.
    nodata_mask = None
    if nodata is not None:
        try:
            nodata_mask = np.all(arr == nodata, axis=0)
        except Exception:
            nodata_mask = None

    # Choose RGB bands.
    if bands >= 3:
        rgb_src = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    elif bands == 2:
        rgb_src = np.stack([arr[0], arr[1], arr[0]], axis=-1)
    else:
        rgb_src = np.stack([arr[0], arr[0], arr[0]], axis=-1)

    # Stretch each channel independently to uint8.
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = _percentile_stretch_to_uint8(rgb_src[..., c])

    return GeoRaster(rgb=rgb, transform=transform, crs=crs, nodata_mask=nodata_mask)


def iter_tiles(
    *,
    case_name: str,
    raster: GeoRaster,
    tile_size: int,
    stride: int,
) -> Iterator[GeoTile]:
    _require_rasterio()
    from affine import Affine

    rgb = raster.rgb
    h, w = rgb.shape[:2]

    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive integers")

    # Ensure last tiles cover the image bounds.
    y_starts = list(range(0, max(1, h - tile_size + 1), stride))
    x_starts = list(range(0, max(1, w - tile_size + 1), stride))
    if not y_starts or y_starts[-1] != max(0, h - tile_size):
        y_starts.append(max(0, h - tile_size))
    if not x_starts or x_starts[-1] != max(0, w - tile_size):
        x_starts.append(max(0, w - tile_size))

    for y0 in y_starts:
        for x0 in x_starts:
            tile = rgb[y0 : y0 + tile_size, x0 : x0 + tile_size]
            # If the raster is smaller than tile_size, pad (rare but safe).
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            # Update transform with pixel offset.
            tile_transform: Affine = raster.transform * Affine.translation(x0, y0)

            yield GeoTile(
                case_name=case_name,
                x0=x0,
                y0=y0,
                rgb=tile,
                transform=tile_transform,
                crs=raster.crs,
                full_height=h,
                full_width=w,
            )

