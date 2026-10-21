"""Compatibility entry point: old commands and run_follower_sweep still work."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from endoscopy_capsule_control.simulation.path_simulation import main

if __name__ == "__main__":
    main()
