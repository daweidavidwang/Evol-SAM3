from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _require_geo_deps():
    try:
        import rasterio  # noqa: F401
        import fiona  # noqa: F401
        import shapely  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Polygon output requires geospatial deps: rasterio + fiona + shapely."
        ) from e


def _crs_to_fiona(crs_obj: object) -> Any:
    # fiona accepts: dict mapping, WKT string, or rasterio CRS.
    if crs_obj is None:
        return None
    # rasterio.crs.CRS has to_wkt()
    to_wkt = getattr(crs_obj, "to_wkt", None)
    if callable(to_wkt):
        return to_wkt()
    return crs_obj


def write_mask_gpkg(
    *,
    mask: np.ndarray,
    transform: "Affine",
    crs: object,
    out_path: str | Path,
    layer_name: str = "fields",
    min_area: float = 0.0,
    simplify_tolerance: Optional[float] = None,
) -> None:
    """
    Polygonize a binary mask (pixel space) into georeferenced polygons and write a GeoPackage.

    - `mask`: bool/0-1 array, shape (H,W). True/1 indicates foreground.
    - `transform`: affine transform mapping pixel coords -> world coords.
    - `crs`: CRS object (rasterio CRS or anything Fiona accepts).
    """
    _require_geo_deps()
    import fiona
    import rasterio.features
    from shapely.geometry import shape, mapping, MultiPolygon, Polygon

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        # Still write an empty layer (useful for downstream batch pipelines).
        schema = {"geometry": "MultiPolygon", "properties": {"id": "int", "area": "float"}}
        with fiona.open(
            out_path,
            mode="w",
            driver="GPKG",
            layer=layer_name,
            crs=_crs_to_fiona(crs),
            schema=schema,
        ):
            return

    geoms = []
    for geom_mapping, value in rasterio.features.shapes(m, mask=m == 1, transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geom_mapping)
        if geom.is_empty:
            continue
        if simplify_tolerance is not None and float(simplify_tolerance) > 0:
            geom = geom.simplify(float(simplify_tolerance), preserve_topology=True)
            if geom.is_empty:
                continue
        if min_area and float(min_area) > 0 and geom.area < float(min_area):
            continue
        if isinstance(geom, Polygon):
            geom = MultiPolygon([geom])
        geoms.append(geom)

    schema = {"geometry": "MultiPolygon", "properties": {"id": "int", "area": "float"}}
    with fiona.open(
        out_path,
        mode="w",
        driver="GPKG",
        layer=layer_name,
        crs=_crs_to_fiona(crs),
        schema=schema,
    ) as dst:
        for idx, geom in enumerate(geoms):
            dst.write(
                {
                    "type": "Feature",
                    "id": str(idx),
                    "geometry": mapping(geom),
                    "properties": {"id": int(idx), "area": float(geom.area)},
                }
            )

