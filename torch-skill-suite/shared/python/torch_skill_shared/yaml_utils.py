"""YAML loading utilities with fallback parser (no PyYAML required)."""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file. Uses PyYAML if available, otherwise a minimal fallback.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary. Returns empty dict if file is missing.
    """
    try:
        if _yaml is not None:
            with open(path, "r", encoding="utf-8") as fh:
                result = _yaml.safe_load(fh)
            return result if isinstance(result, dict) else {}
        return _fallback_load(path)
    except (FileNotFoundError, OSError):
        return {}


def _fallback_load(path: str) -> Dict[str, Any]:
    """Minimal YAML parser for simple key-value configs."""
    result: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val in ("true", "True"):
                val = True
            elif val in ("false", "False"):
                val = False
            elif val == "null":
                val = None
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        val = val.strip("'\"")
            result[key] = val
    return result


def find_shared_file(filename: str, start_dir: Optional[str] = None) -> Optional[str]:
    """Find a shared file by walking up from start_dir to the project root.

    Args:
        filename: The filename to find (e.g., 'data_contract.schema.json').
        start_dir: Directory to start searching from (default: cwd).

    Returns:
        Absolute path to the file, or None if not found.
    """
    root_markers = ["CLAUDE.md", "CLAUDE_ZH.md", ".git"]
    current = Path(start_dir or ".").resolve()

    while True:
        # Check if we're at the project root
        if any((current / m).exists() for m in root_markers):
            # Search shared/schemas and shared/contracts
            for subdir in ["shared/schemas", "shared/contracts"]:
                candidate = current / subdir / filename
                if candidate.exists():
                    return str(candidate.resolve())
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
