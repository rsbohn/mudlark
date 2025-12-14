import json
from pathlib import Path
from typing import Dict, Optional


class Mapper:
    """Best-effort room graph builder that writes mudmap.json."""

    def __init__(self, map_path: Optional[Path] = None):
        self.map_path = map_path or Path("mudmap.json")
        self.map_data: Dict = {"rooms": {}}
        self.last_move_dir: Optional[str] = None
        self.directions = {"n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw",
                           "north", "south", "east", "west", "up", "down", "in", "out"}
        self.dir_aliases = {
            "north": "n", "south": "s", "east": "e", "west": "w",
            "up": "u", "down": "d",
            "northeast": "ne", "northwest": "nw",
            "southeast": "se", "southwest": "sw",
            "in": "in", "out": "out"
        }
        self.load_map()

    def load_map(self) -> None:
        """Load existing map file if present."""
        try:
            if self.map_path.exists():
                data = json.loads(self.map_path.read_text())
                if isinstance(data, dict) and "rooms" in data:
                    self.map_data = data
                else:
                    self.map_data = {"rooms": {}}
        except Exception:
            self.map_data = {"rooms": {}}

    def save_map(self) -> None:
        """Persist map to disk (best-effort)."""
        try:
            self.map_path.write_text(json.dumps(self.map_data, indent=2, sort_keys=True))
        except Exception:
            pass

    def register_move_command(self, message: str) -> None:
        """Track a potential movement command so we can map the next room change."""
        cmd = message.strip().lower()
        if cmd in self.directions:
            self.last_move_dir = self.dir_aliases.get(cmd, cmd)
        else:
            self.last_move_dir = None

    def record_transition(self, origin: str, direction: str, dest: str) -> None:
        """Record a directional edge between rooms."""
        rooms = self.map_data.setdefault("rooms", {})
        origin_entry = rooms.setdefault(origin, {"exits": {}})
        dest_entry = rooms.setdefault(dest, {"exits": {}})
        origin_entry["exits"][direction] = dest
        rooms[origin] = origin_entry
        rooms[dest] = dest_entry
        self.save_map()

    def handle_location_change(self, origin: Optional[str], dest: Optional[str]) -> None:
        """Hook called on new room; will map the last movement if known."""
        if origin and dest and self.last_move_dir:
            self.record_transition(origin, self.last_move_dir, dest)
        self.last_move_dir = None
