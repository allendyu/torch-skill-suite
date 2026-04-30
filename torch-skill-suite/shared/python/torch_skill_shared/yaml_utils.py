"""YAML loading/emitting utilities with fallback parser (no PyYAML required)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


# ---------------------------------------------------------------------------
# SimpleYAMLParser — full nested YAML parser (PyYAML fallback)
# ---------------------------------------------------------------------------

class SimpleYAMLParser:
    """Minimal YAML parser supporting nested dicts, lists, inline comments.

    Handles the subset of YAML used by contract files: scalars, nested maps,
    sequences, inline comments, quoted strings, and JSON-like literals.
    """

    def __init__(self, text: str):
        self.lines = self._prepare_lines(text)

    def _strip_inline_comment(self, line: str) -> str:
        in_single, in_double = False, False
        escaped = False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if ch == "#" and not in_single and not in_double:
                if i == 0 or line[i - 1].isspace():
                    return line[:i].rstrip()
        return line.rstrip()

    def _prepare_lines(self, text: str):
        return [self._strip_inline_comment(line) for line in text.splitlines()]

    def parse(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        stack = [(None, 0, result)]
        for line in self.lines:
            content = line.rstrip()
            if not content or content.isspace():
                continue
            indent = len(line) - len(line.lstrip())
            key, value = self._parse_line(content)
            if key is None:
                continue
            while stack and stack[-1][1] >= indent:
                stack.pop()
            parent = stack[-1][2]
            if value is not None:
                parent[key] = value
            else:
                new: Dict[str, Any] = {}
                parent[key] = new
                stack.append((key, indent, new))
        return result

    def _parse_line(self, line: str):
        content = line.strip()
        if content.startswith("- "):
            content = content[2:]
            key = None
        elif ":" in content:
            parts = content.split(":", 1)
            key = parts[0].strip().strip("'\"")
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                return key, None
            content = rest
        else:
            return None, None
        return key, self._parse_value(content)

    def _parse_value(self, text: str):
        if text in ("true", "True", "yes"):
            return True
        if text in ("false", "False", "no"):
            return False
        if text in ("null", "None", "~"):
            return None
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            return self._parse_json_like(text)
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        try:
            return int(text)
        except (ValueError, TypeError):
            pass
        try:
            return float(text)
        except (ValueError, TypeError):
            pass
        return text

    def _parse_json_like(self, text: str):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return self._parse_loose_list(text)

    def _parse_loose_list(self, text: str):
        text = text.strip("[] ")
        if not text:
            return []
        items = []
        depth = 0
        current = ""
        for ch in text:
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth -= 1
            if ch == "," and depth == 0:
                items.append(self._parse_value(current.strip()))
                current = ""
            else:
                current += ch
        if current.strip():
            items.append(self._parse_value(current.strip()))
        return items


# ---------------------------------------------------------------------------
# YAML I/O helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file. Uses PyYAML if available, otherwise SimpleYAMLParser.

    Returns empty dict if file is missing or unreadable.
    """
    try:
        if _yaml is not None:
            with open(path, "r", encoding="utf-8") as fh:
                result = _yaml.safe_load(fh)
            return result if isinstance(result, dict) else {}
        with open(path, "r", encoding="utf-8") as fh:
            return SimpleYAMLParser(fh.read()).parse()
    except (FileNotFoundError, OSError):
        return {}


def emit_yaml(data: Any) -> str:
    """Emit a dict/list as YAML. Uses PyYAML safe_dump if available, otherwise fallback."""
    if _yaml is not None:
        return _yaml.safe_dump(data, default_flow_style=False, allow_unicode=True)
    return _emit_yaml_fallback(data)


def _emit_yaml_fallback(data: Any, indent: int = 0) -> str:
    """Minimal YAML emitter for dicts, lists, and scalars."""
    lines = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_emit_yaml_fallback(value, indent + 1))
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                if any(ch in value for ch in ": #{}[]&*!|>'\"%@`"):
                    lines.append(f"{prefix}{key}: '{value}'")
                else:
                    lines.append(f"{prefix}{key}: {value}")
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            elif isinstance(value, float):
                lines.append(f"{prefix}{key}: {value}")
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                inner = _emit_yaml_fallback(item, indent + 1).lstrip()
                lines.append(f"{prefix}- {inner}")
            elif isinstance(item, bool):
                lines.append(f"{prefix}- {'true' if item else 'false'}")
            elif isinstance(item, str):
                lines.append(f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}- {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

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
        if any((current / m).exists() for m in root_markers):
            for subdir in ["shared/schemas", "shared/contracts"]:
                candidate = current / subdir / filename
                if candidate.exists():
                    return str(candidate.resolve())
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent
