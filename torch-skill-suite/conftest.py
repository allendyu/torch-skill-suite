"""Root conftest: add shared python package to sys.path."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent

# Add shared python package for all tests
_shared_pkg = _root / "shared" / "python"
if _shared_pkg.exists() and str(_shared_pkg) not in sys.path:
    sys.path.insert(0, str(_shared_pkg))
