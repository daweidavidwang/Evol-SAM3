from __future__ import annotations

import os
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.dataset import get_dataset_loader
from src.geo_preprocess import read_geotiff_as_rgb, iter_tiles
from src.solver import Solver
from src.utils import ExperimentLogger, save_checkpoint, load_checkpoint


def _get_nested(cfg, path: str, default):
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 95) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    b = BytesIO()
    img.save(b, format="JPEG", quality=int(quality))
    return b.getvalue()


class FBISEvaluator:
    """
    Evol-SAM3 harness for FBIS-22M GeoTIFFs.

    - Reads paths from `dataset.list_path`
    - Uses a fixed prompt from `dataset.prompt`
    - Runs the existing `Solver` to produce a binary mask
    - Polygonization + GPKG writing is handled by `src.polygonize`
    """

    def __init__(self, cfg, qwen_engine, sam_engine):
        self.cfg = cfg
        self.qwen = qwen_engine
        self.sam = sam_engine

        self.log_dir = cfg.paths.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.log_dir, "checkpoint_fbis.json")

        # Output root for GPKGs (separate from log_dir).
        self.output_root = _get_nested(cfg, "output.output_root", None)
        if not self.output_root:
            self.output_root = os.path.join(self.log_dir, "fbis_outputs")
        os.makedirs(self.output_root, exist_ok=True)

        self.prompt = _get_nested(cfg, "dataset.prompt", None)
        if not self.prompt or not str(self.prompt).strip():
            raise ValueError("FBIS requires `dataset.prompt` (fixed text prompt) in the config.")

        self.tiling_enabled = bool(_get_nested(cfg, "dataset.tiling.enabled", True))
        self.tile_size = int(_get_nested(cfg, "dataset.tiling.tile_size", 1024))
        self.stride = int(_get_nested(cfg, "dataset.tiling.stride", self.tile_size))

        self.jpeg_quality = int(_get_nested(cfg, "dataset.jpeg_quality", 95))

    def _already_done(self, case_name: str) -> bool:
        out_path = os.path.join(self.output_root, f"{case_name}.gpkg")
        try:
            return os.path.exists(out_path) and os.path.getsize(out_path) > 0
        except OSError:
            return False

    def run(self):
        loader = get_dataset_loader(self.cfg)
        dataset = loader.data

        start_index = 0
        if os.path.exists(self.ckpt_path):
            start_index = load_checkpoint(self.ckpt_path, meters_dict={})

        pbar = tqdm(range(start_index, len(dataset)), desc="FBIS (Evol-SAM3)")
        for i in pbar:
            item = dataset[i]
            tif_path = item["tif_path"]
            case_name = item["case_name"]

            pbar.set_postfix({"case": case_name})

            if self._already_done(case_name):
                save_checkpoint(self.ckpt_path, i, meters_dict={}, split="fbis")
                continue

            case_logger = ExperimentLogger(self.log_dir, case_name, resume=False)
            case_logger.log("Data", f"Index [{i}/{len(dataset)}] tif={tif_path}")
            case_logger.log("Data", f"Prompt: {self.prompt}")

            raster = read_geotiff_as_rgb(tif_path)
            h, w = raster.rgb.shape[:2]

            full_mask = np.zeros((h, w), dtype=bool)

            if self.tiling_enabled:
                tiles = list(iter_tiles(case_name=case_name, raster=raster, tile_size=self.tile_size, stride=self.stride))
            else:
                from dataclasses import replace

                tiles = [
                    replace(
                        next(iter_tiles(case_name=case_name, raster=raster, tile_size=h, stride=h)),
                        x0=0,
                        y0=0,
                        full_height=h,
                        full_width=w,
                    )
                ]

            for t_idx, tile in enumerate(tiles):
                tile_tag = f"{case_name}_x{tile.x0}_y{tile.y0}"
                case_logger.log("Tile", f"[{t_idx+1}/{len(tiles)}] {tile_tag}")

                img_bytes = _rgb_to_jpeg_bytes(tile.rgb, quality=self.jpeg_quality)

                solver = Solver(
                    cfg=self.cfg,
                    img_bytes=img_bytes,
                    query=str(self.prompt),
                    logger=case_logger,
                    mllm_engine=self.qwen,
                    sam_engine=self.sam,
                    fname=tile_tag,
                )
                final_ind = solver.run()
                if not final_ind or final_ind.M is None:
                    continue

                tile_mask = final_ind.M
                while tile_mask.ndim > 2:
                    tile_mask = tile_mask.squeeze(0)
                tile_mask = tile_mask.astype(bool)

                hh = min(tile_mask.shape[0], h - tile.y0)
                ww = min(tile_mask.shape[1], w - tile.x0)
                if hh <= 0 or ww <= 0:
                    continue
                full_mask[tile.y0 : tile.y0 + hh, tile.x0 : tile.x0 + ww] |= tile_mask[:hh, :ww]

            # Polygonize + write GPKG (implemented in src/polygonize.py).
            try:
                from src.polygonize import write_mask_gpkg
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Polygon output requires `src/polygonize.py` and geospatial dependencies."
                ) from e

            out_path = os.path.join(self.output_root, f"{case_name}.gpkg")
            write_mask_gpkg(
                mask=full_mask,
                transform=raster.transform,
                crs=raster.crs,
                out_path=out_path,
                layer_name=_get_nested(self.cfg, "output.layer_name", "fields"),
                min_area=float(_get_nested(self.cfg, "output.min_area", 0.0)),
                simplify_tolerance=_get_nested(self.cfg, "output.simplify_tolerance", None),
            )

            case_logger.log("Output", f"Wrote: {out_path}")

            save_checkpoint(self.ckpt_path, i, meters_dict={}, split="fbis")

