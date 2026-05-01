"""
python/zombiebubble/_core.py

Loads the compiled Zig shared library (libzombiebubble.so / zombiebubble.dll /
libzombiebubble.dylib) and declares all ctypes signatures.

The library is looked up next to this file first, then on the system library
path.  Build it with:

    zig build pip-lib
    cp zig-out/lib/libzombiebubble.* python/zombiebubble/

or let the setuptools build-hook do it (see pyproject.toml).
"""

import ctypes
import pathlib
import platform
import sys

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_lib() -> ctypes.CDLL:
    here = pathlib.Path(__file__).parent
    system = platform.system()
    if system == "Windows":
        candidates = ["zombiebubble.dll"]
    elif system == "Darwin":
        candidates = ["libzombiebubble.dylib"]
    else:
        candidates = ["libzombiebubble.so"]

    for name in candidates:
        candidate = here / name
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    # Fall back to system search (useful when installed site-wide)
    try:
        return ctypes.CDLL(candidates[0])
    except OSError:
        raise FileNotFoundError(
            f"zombiebubble shared library not found ({', '.join(candidates)}).\n"
            "Run 'zig build pip-lib' then copy the output into python/zombiebubble/."
        ) from None


_lib = _find_lib()

# ---------------------------------------------------------------------------
# C type aliases
# ---------------------------------------------------------------------------

_f32 = ctypes.c_float
_u32 = ctypes.c_uint32
_ptr_f32 = ctypes.POINTER(ctypes.c_float)
_void_p = ctypes.c_void_p
_ptr_f32_out = ctypes.POINTER(ctypes.c_float)

# ---------------------------------------------------------------------------
# Swarm lifecycle
# ---------------------------------------------------------------------------

_lib.zb_swarm_create.restype = _void_p
_lib.zb_swarm_create.argtypes = [_u32]

_lib.zb_swarm_destroy.restype = None
_lib.zb_swarm_destroy.argtypes = [_void_p]

_lib.zb_swarm_count.restype = _u32
_lib.zb_swarm_count.argtypes = [_void_p]

# ---------------------------------------------------------------------------
# Config setters
# ---------------------------------------------------------------------------

for _name in (
    "zb_swarm_set_cohesion_weight",
    "zb_swarm_set_alignment_weight",
    "zb_swarm_set_separation_weight",
    "zb_swarm_set_max_speed",
    "zb_swarm_set_cohesion_radius",
    "zb_swarm_set_separation_radius",
):
    fn = getattr(_lib, _name)
    fn.restype = None
    fn.argtypes = [_void_p, _f32]

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_lib.zb_swarm_step.restype = None
_lib.zb_swarm_step.argtypes = [_void_p, _f32]

_lib.zb_swarm_get_positions.restype = None
_lib.zb_swarm_get_positions.argtypes = [_void_p, _ptr_f32_out, _u32]

_lib.zb_swarm_get_velocities.restype = None
_lib.zb_swarm_get_velocities.argtypes = [_void_p, _ptr_f32_out, _u32]

# ---------------------------------------------------------------------------
# Vec3 math
# ---------------------------------------------------------------------------

_lib.zb_vec3_dot.restype = _f32
_lib.zb_vec3_dot.argtypes = [_f32] * 6

_lib.zb_vec3_length.restype = _f32
_lib.zb_vec3_length.argtypes = [_f32] * 3

for _name in ("zb_vec3_add", "zb_vec3_sub", "zb_vec3_cross"):
    fn = getattr(_lib, _name)
    fn.restype = None
    fn.argtypes = [_f32] * 6 + [ctypes.POINTER(_f32)] * 3

_lib.zb_vec3_scale.restype = None
_lib.zb_vec3_scale.argtypes = [_f32, _f32, _f32, _f32] + [ctypes.POINTER(_f32)] * 3

_lib.zb_vec3_lerp.restype = None
_lib.zb_vec3_lerp.argtypes = [_f32] * 7 + [ctypes.POINTER(_f32)] * 3

_lib.zb_vec3_normalize.restype = None
_lib.zb_vec3_normalize.argtypes = [_f32] * 3 + [ctypes.POINTER(_f32)] * 3

# ---------------------------------------------------------------------------
# Convenience: Vec3 Python helpers
# ---------------------------------------------------------------------------

def _v3_out():
    """Return a tuple (ox, oy, oz) of c_float refs and a reader closure."""
    ox, oy, oz = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
    return ox, oy, oz, lambda: (ox.value, oy.value, oz.value)


def vec3_add(a, b):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_add(a[0], a[1], a[2], b[0], b[1], b[2], ox, oy, oz)
    return read()


def vec3_sub(a, b):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_sub(a[0], a[1], a[2], b[0], b[1], b[2], ox, oy, oz)
    return read()


def vec3_scale(v, s):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_scale(v[0], v[1], v[2], s, ox, oy, oz)
    return read()


def vec3_dot(a, b):
    return _lib.zb_vec3_dot(a[0], a[1], a[2], b[0], b[1], b[2])


def vec3_cross(a, b):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_cross(a[0], a[1], a[2], b[0], b[1], b[2], ox, oy, oz)
    return read()


def vec3_lerp(a, b, t):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_lerp(a[0], a[1], a[2], b[0], b[1], b[2], t, ox, oy, oz)
    return read()


def vec3_normalize(v):
    ox, oy, oz, read = _v3_out()
    _lib.zb_vec3_normalize(v[0], v[1], v[2], ox, oy, oz)
    return read()


def vec3_length(v):
    return _lib.zb_vec3_length(v[0], v[1], v[2])
