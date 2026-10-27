"""Parsing helpers for importing mapped objects from screen-read coordinates."""
import re


_LABELED = re.compile(
    r"X\s*([+-]?\d+(?:\.\d+)?)\s+Y\s*([+-]?\d+(?:\.\d+)?)\s+Z\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_xyz_points(text):
    """Parse one XYZ point per line, accepting labeled or bare inch values."""
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("mapped "):
            continue
        labeled = _LABELED.search(line)
        if labeled:
            points.append([float(value) for value in labeled.groups()])
            continue
        values = re.findall(r"[+-]?\d+(?:\.\d+)?", line)
        if len(values) >= 4 and re.match(r"\s*\d+\s*[.)]", line):
            values = values[1:]
        if len(values) != 3:
            raise ValueError(f"Could not read point: {line}")
        points.append([float(value) for value in values])
    return points
