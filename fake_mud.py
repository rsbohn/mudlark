#!/usr/bin/env python3
"""
Tiny fake MUD responder for testing Mudlark.

Run with socat, e.g.:
  socat -v TCP-LISTEN:4000,reuseaddr,fork EXEC:"python fake_mud.py"
Then connect Mudlark to localhost 4000:
  uv run main.py localhost 4000
"""

import sys


ROOMS = [
    """<stats>w
The Dark Alley
The dark alley. Much like any other location.

      The body of the beastly fido lies here, sliced and diced.
      A puddle of water has formed here.

Obvious exits:
North: Another room
South: Another room""",
    """<stats>e
Common Square
A small square with a fountain in the center. People mill about.

      A wooden bench sits near the fountain.

Obvious exits:
North: Another room
South: Another room""",
]


def send_room(idx: int) -> None:
    """Send a room block to the client."""
    sys.stdout.write(ROOMS[idx] + "\r\n")
    sys.stdout.flush()


def main() -> None:
    idx = 0
    send_room(idx)
    for raw in sys.stdin:
        cmd = raw.strip().lower()
        if cmd in {"q", "quit", "exit"}:
            sys.stdout.write("Goodbye!\r\n")
            sys.stdout.flush()
            break
        if cmd in {"n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw", "look", "l", ""}:
            idx = (idx + 1) % len(ROOMS)
            send_room(idx)
        else:
            sys.stdout.write(f"You said: {raw}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
