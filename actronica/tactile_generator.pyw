"""Double-click to launch Tactile Lab Generator (or: python tactile_generator.pyw)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tactile_generator.app import main  # noqa: E402

main()
