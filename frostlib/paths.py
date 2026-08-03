"""Data locations and the raw-CSV naming convention.

``fetch_data.py`` writes the raw files and ``prepare.py`` reads them back, so the
filename format is a contract between the two: it is defined once here, together
with the parser that recovers the station name from it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Raw ISD downloads (gitignored; a symlink to a larger filesystem in practice).
RAW_DIR = REPO_ROOT / "data_raw"
RAW_GLOB = "isd_*.csv"

DATA_DIR = REPO_ROOT / "data"
NIGHTS_NAME = "nights.csv"
METRICS_NAME = "metrics.json"
MODEL_NAME = "model.joblib"
NIGHTS_CSV = DATA_DIR / NIGHTS_NAME
METRICS_JSON = DATA_DIR / METRICS_NAME
MODEL_PATH = DATA_DIR / MODEL_NAME


def raw_csv_path(station_name: str, station_id: str, year: int,
                 raw_dir: Path = RAW_DIR) -> Path:
    """Where one raw station-year CSV lives."""
    return raw_dir / f"isd_{station_name}_{station_id}_{year}.csv"


def station_name_from_path(csv_path: Path) -> str:
    """isd_<name>_<id>_<year>.csv -> <name>.

    removeprefix (not replace) so an "isd_" substring elsewhere in the name
    cannot be mangled.
    """
    return csv_path.stem.removeprefix("isd_").rsplit("_", 2)[0]


def raw_station_years(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Every raw station-year CSV present, in a stable order."""
    return sorted(raw_dir.glob(RAW_GLOB))
