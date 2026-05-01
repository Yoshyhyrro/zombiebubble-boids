"""
python/zombiebubble/simulation.py

High-level Python wrapper around the Zig boid simulation.
"""

from __future__ import annotations

import ctypes
from typing import Sequence

from . import _core


class Swarm:
    """
    A boid swarm backed by the compiled Zig simulation engine.

    Parameters
    ----------
    count:
        Number of boids (1–4096).

    The swarm is initialised with boids scattered on a unit circle in the XZ
    plane, all at rest.  Call :meth:`step` once per simulation tick.
    """

    def __init__(self, count: int) -> None:
        if count < 1 or count > 4096:
            raise ValueError(f"count must be 1–4096, got {count}")
        handle = _core._lib.zb_swarm_create(count)
        if not handle:
            raise RuntimeError(f"zb_swarm_create({count}) returned NULL")
        self._handle = handle
        self._count = count
        # Reusable flat buffer: [x0,y0,z0, x1,y1,z1, ...]
        self._buf = (ctypes.c_float * (count * 3))()

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _core._lib.zb_swarm_destroy(self._handle)
            self._handle = None

    # ------------------------------------------------------------------ step

    def step(self, dt: float = 0.016) -> None:
        """Advance the simulation by *dt* seconds."""
        _core._lib.zb_swarm_step(self._handle, dt)

    # ------------------------------------------------------------------ read

    def positions(self) -> list[tuple[float, float, float]]:
        """
        Return a list of (x, y, z) tuples — one per boid.
        Allocates a new Python list each call; for tight loops prefer
        :meth:`positions_into`.
        """
        self._fill_buf(_core._lib.zb_swarm_get_positions)
        return self._buf_to_list()

    def velocities(self) -> list[tuple[float, float, float]]:
        """Return a list of (vx, vy, vz) tuples — one per boid."""
        self._fill_buf(_core._lib.zb_swarm_get_velocities)
        return self._buf_to_list()

    def positions_into(self, out: ctypes.Array) -> None:
        """
        Fill a pre-allocated ``(c_float * (count * 3))`` array in-place.
        Zero-copy path for NumPy integration::

            buf = (ctypes.c_float * (swarm.count * 3))()
            swarm.positions_into(buf)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(-1, 3)
        """
        _core._lib.zb_swarm_get_positions(self._handle, out, self._count * 3)

    def velocities_into(self, out: ctypes.Array) -> None:
        """Fill a pre-allocated ``(c_float * (count * 3))`` array in-place."""
        _core._lib.zb_swarm_get_velocities(self._handle, out, self._count * 3)

    # ---------------------------------------------------------------- config

    @property
    def count(self) -> int:
        return self._count

    def set_cohesion_weight(self, w: float) -> None:
        _core._lib.zb_swarm_set_cohesion_weight(self._handle, w)

    def set_alignment_weight(self, w: float) -> None:
        _core._lib.zb_swarm_set_alignment_weight(self._handle, w)

    def set_separation_weight(self, w: float) -> None:
        _core._lib.zb_swarm_set_separation_weight(self._handle, w)

    def set_max_speed(self, v: float) -> None:
        _core._lib.zb_swarm_set_max_speed(self._handle, v)

    def set_cohesion_radius(self, r: float) -> None:
        _core._lib.zb_swarm_set_cohesion_radius(self._handle, r)

    def set_separation_radius(self, r: float) -> None:
        _core._lib.zb_swarm_set_separation_radius(self._handle, r)

    # ------------------------------------------------------------ internals

    def _fill_buf(self, fn) -> None:
        fn(self._handle, self._buf, self._count * 3)

    def _buf_to_list(self) -> list[tuple[float, float, float]]:
        b = self._buf
        return [(b[i * 3], b[i * 3 + 1], b[i * 3 + 2]) for i in range(self._count)]
