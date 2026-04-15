from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FBISItem:
    tif_path: Path
    case_name: str


def _make_unique_name(base_name: str, used_names: dict[str, int]) -> str:
    count = used_names.get(base_name, 0)
    used_names[base_name] = count + 1
    if count == 0:
        return base_name
    return f"{base_name}_{count}"


def _resolve_image_path(raw_path: str, test_list_path: Path) -> Path:
    # Mirrors Delineate-Anything/run_fbis_test_batch.py behavior.
    line_path = Path(raw_path)
    candidates: list[Path] = [line_path]

    candidates.append(test_list_path.parent / line_path)
    candidates.append(test_list_path.parent.parent / line_path)

    prefix = "FBIS-22M/"
    if raw_path.startswith(prefix):
        trimmed = raw_path[len(prefix) :]
        candidates.append(test_list_path.parent / trimmed)
        candidates.append(test_list_path.parent.parent / trimmed)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve image path from list entry: {raw_path}")


class FBISDatasetLoader:
    """
    List-based loader for FBIS-22M GeoTIFF imagery.

    Contract:
    - `list_path` is a txt file with one path per line (absolute or relative).
    - produces `self.data` list of dicts with keys:
        - 'tif_path': str
        - 'case_name': str
    """

    def __init__(self, list_path: str | Path):
        self.list_path = Path(list_path).expanduser().resolve()
        self.data: list[dict] = []
        self._load()

    def _iter_lines(self) -> Iterable[str]:
        txt = self.list_path.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            yield line

    def _load(self) -> None:
        if not self.list_path.exists():
            raise FileNotFoundError(f"FBIS list file not found: {self.list_path}")

        used_names: dict[str, int] = {}
        for raw in self._iter_lines():
            tif_path = _resolve_image_path(raw, self.list_path)
            case_name = _make_unique_name(tif_path.stem, used_names)
            self.data.append(
                {
                    "tif_path": str(tif_path),
                    "case_name": case_name,
                }
            )

