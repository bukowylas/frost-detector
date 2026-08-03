"""Put the repository root on sys.path so tests can import the pipeline modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
