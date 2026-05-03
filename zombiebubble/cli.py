"""
python/zombiebubble/cli.py

Entry point for the `zombiebubble` command-line tool.
"""

from __future__ import annotations

import argparse
import ast
import math
import pathlib
import sys
from dataclasses import dataclass
from enum import Enum

# ── Boids intro text ──────────────────────────────────────────────────────────

_BOIDS_INTRO = """\
What are Boids?
───────────────
Boids (bird-oids) simulates how animals flock together — like starlings
murmurating at dusk, or fish schooling in the ocean.

Each agent follows just 3 simple local rules.  No central coordinator exists;
the flock emerges spontaneously from individual behaviour:

  COHESION    Steer toward the average position of nearby neighbours.
              → keeps the group from drifting apart

  ALIGNMENT   Match the average velocity of nearby neighbours.
              → the whole flock starts flying in the same direction

  SEPARATION  Steer away from neighbours that get too close.
              → prevents collisions and crowding
"""

# ── Kernel diagnostics ───────────────────────────────────────────────────────
#
# All kernel parameters are internal.  User-visible messages use behavioural
# language only.  Diagnostic codes are back-calculated from the linear-code
# design in the following severity ladder:
#
#   BW001  error    N < d      correction graph cannot form
#   BW002  warning  N < k      election underconstrained
#   BW003  warning  N < n      weight-class coverage incomplete
#   BW004  note     N ≡ n−1 (mod n), N < n
#                              structural complement gap
#                              (f9 / Frobenius-Schur obstruction analog:
#                               prime-leader w_{n-1} loses its isolated
#                               complement w_0 in the 6-class HN kernel)
#   BW005  note     N ≡ q−1 (mod q), N < n
#                              phase-group boundary
#                              (adds_topology / forgets_topology parity break
#                               in the HN depth tower)
#          note     N < erasure_margin  →  erasure-margin suggestion

_K              = {"n": 7, "k": 5, "d": 3, "q": 5}
_T              = (_K["d"] - 1) // 2          # error-correction capacity = 1
_ERASURE        = _K["d"] - 1                 # erasure capacity           = 2
_ERASURE_MARGIN = _K["n"] + 2 * _T           # erasure-safe minimum       = 9


class _Level(Enum):
    ERROR   = "error"
    WARNING = "warning"
    NOTE    = "note"


@dataclass(frozen=True)
class _Diag:
    level:   _Level
    code:    str | None   # e.g. "BW001"; None for codeless notes
    message: str
    help:    str | None = None


def _diagnose(n_boids: int) -> list[_Diag]:
    """Return diagnostics for n_boids, most-severe first."""
    n, k, d, q = _K["n"], _K["k"], _K["d"], _K["q"]
    out: list[_Diag] = []

    # ── Primary threshold ────────────────────────────────────────────────────
    if n_boids < d:
        out.append(_Diag(
            _Level.ERROR, "BW001",
            f"{n_boids} boids < {d} (min-distance d): "
            "error-correction structure collapsed",
            f"the kernel corrects t=⌊(d−1)/2⌋={_T} simultaneous leader loss "
            f"but needs ≥ d={d} boids to form the correction graph; "
            f"add at least {d - n_boids} more",
        ))
    elif n_boids < k:
        out.append(_Diag(
            _Level.WARNING, "BW002",
            f"{n_boids} boids < {k} (degrees of freedom k): "
            "election underconstrained",
            f"leader election needs k={k} independent signal dimensions; "
            "the flock may lock into a degenerate low-rank configuration",
        ))
    elif n_boids < n:
        out.append(_Diag(
            _Level.WARNING, "BW003",
            f"{n_boids} boids < {n} (weight classes n): "
            "weight-class coverage incomplete",
            f"the scoring kernel has n={n} roles; with fewer boids the "
            "prime-leader role (highest class) may never be elected",
        ))

    # ── BW004: structural complement gap (Frobenius-Schur obstruction analog) ─
    # In the 6-class HN-depth kernel [6,5,3]_5 the isolated role w0 is
    # collapsed.  A count N ≡ n−1 (mod n) fills exactly n−1 classes, leaving
    # w0 empty while w_{n-1} (prime-leader) is populated — its complement
    # partner is missing.  Directly analogous to FischerCarabiner f9:
    #   h(f9) + h(f9) = 18 ≠ 12  because the height-3 partner does not exist
    #   in the ternary Golay weight set.
    if d <= n_boids < n and n_boids % n == n - 1:
        out.append(_Diag(
            _Level.NOTE, "BW004",
            f"structural complement gap: {n_boids} ≡ {n - 1} (mod {n}) — "
            "prime-leader may be elected with no isolated complement (w0 absent)",
            f"add 1 boid (→ {n_boids + 1}) to close the complement pair",
        ))

    # ── BW005: phase-group boundary ──────────────────────────────────────────
    # In the q=5 phase group the band N ≡ q−1 (mod q) falls on the
    # adds_topology / forgets_topology parity boundary in the HN depth tower.
    # When this coincides with incomplete weight coverage, complement-dual
    # height closure is unreliable.
    if d <= n_boids < n and n_boids % q == q - 1:
        out.append(_Diag(
            _Level.NOTE, "BW005",
            f"phase-group boundary: {n_boids} ≡ {q - 1} (mod {q}) — "
            "weight assignment falls on the parity-breaking depth level; "
            "complement-dual height may not close",
            f"add 1 boid (→ {n_boids + 1}) to step off the phase boundary",
        ))

    # ── Erasure-margin suggestion ─────────────────────────────────────────────
    if d <= n_boids < _ERASURE_MARGIN:
        out.append(_Diag(
            _Level.NOTE, None,
            f"for full erasure-correction margin "
            f"(d−1={_ERASURE} simultaneous leader losses tolerated), "
            f"recommend ≥ {_ERASURE_MARGIN} boids",
        ))

    return out

def _fmt_run(diag: _Diag) -> str:
    """Format a diagnostic for `run` (no source context)."""
    tag = f"{diag.level.value}[{diag.code}]" if diag.code else diag.level.value
    lines = [f"{tag}: {diag.message}"]
    if diag.help:
        lines.append(f"  = help: {diag.help}")
    return "\n".join(lines)


def _fmt_check(
    diag: _Diag,
    path: str,
    lineno: int,
    col: int,
    source_line: str,
    token_len: int,
) -> str:
    """Format a diagnostic in rustc style with source context."""
    tag = f"{diag.level.value}[{diag.code}]" if diag.code else diag.level.value
    gw  = len(str(lineno))
    pad = " " * gw
    lines = [
        f"{tag}: {diag.message}",
        f"  --> {path}:{lineno}:{col}",
        f"   {pad}|",
        f"{lineno:>{gw}} | {source_line.rstrip()}",
        f"   {pad}| {' ' * (col - 1)}{'~' * token_len}",
        f"   {pad}|",
    ]
    if diag.help:
        lines.append(f"   {pad}= help: {diag.help}")
    return "\n".join(lines)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _spread(positions: list[tuple[float, float, float]]) -> float:
    """Average distance from centroid — smaller means a tighter flock."""
    n = len(positions)
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    return sum(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
               for p in positions) / n


def _avg_speed(velocities: list[tuple[float, float, float]]) -> float:
    return sum(math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in velocities) / len(velocities)


def _flock_label(spread: float, speed: float, initial_spread: float) -> str:
    if speed < 0.05:
        return "at rest"
    if spread > initial_spread * 0.95:
        return "scattered — starting to sense neighbours"
    if spread > initial_spread * 0.70:
        return "cohesion kicking in — flock pulling together"
    if spread > initial_spread * 0.50:
        return "aligning — agents flying in similar directions"
    return "tight formation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zombiebubble",
        description="ZombieBubble ECS boids simulation (free tier).",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    # ---- zombiebubble run ---------------------------------------------------
    run_p = sub.add_parser("run", help="Run a headless boids simulation.")
    run_p.add_argument("--boids", type=int, default=20,
                       metavar="N", help="Number of boids (default: 20)")
    run_p.add_argument("--steps", type=int, default=200,
                       metavar="S", help="Simulation steps (default: 200)")
    run_p.add_argument("--dt", type=float, default=0.016,
                       metavar="DT", help="Time step in seconds (default: 0.016)")
    run_p.add_argument("--print-every", type=int, default=40,
                       metavar="K", help="Print metrics every K steps (default: 40)")
    run_p.add_argument("--no-intro", action="store_true",
                       help="Skip the Boids explanation")

    # ---- zombiebubble info --------------------------------------------------
    sub.add_parser("info", help="Show library build info.")

    # ---- zombiebubble check -------------------------------------------------
    check_p = sub.add_parser(
        "check",
        help="Statically check a Python file for risky Swarm() call sites.",
        description=(
            "Parse a Python source file and warn about Swarm(N) calls whose N "
            "falls below the kernel thresholds for leader election and flock "
            "expressiveness.  Exit code 1 if any warnings are emitted."
        ),
    )
    check_p.add_argument(
        "file",
        metavar="FILE",
        help="Python source file to analyse (e.g. my_sim.py)",
    )
    check_p.add_argument(
        "--error",
        action="store_true",
        help="Treat threshold warnings as errors (exit 2)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args)
    elif args.cmd == "info":
        return _cmd_info()
    elif args.cmd == "check":
        return _cmd_check(args)
    else:
        parser.print_help()
        return 0


def _cmd_run(args) -> int:
    from .simulation import Swarm

    n, steps, dt, every = args.boids, args.steps, args.dt, args.print_every

    if n < 1 or n > 4096:
        print("error: --boids must be 1–4096", file=sys.stderr)
        return 1

    for diag in _diagnose(n):
        print(_fmt_run(diag), file=sys.stderr)

    if not args.no_intro:
        print(_BOIDS_INTRO)

    total_time = steps * dt
    print(f"Simulation: {n} boids × {steps} steps × dt={dt}s  "
          f"(≈{total_time:.1f}s simulated time)")
    print()

    header = f"{'step':>5}  {'time':>6}  {'spread':>8}  {'avg speed':>9}  status"
    print(header)
    print("─" * len(header))

    swarm = Swarm(n)
    initial_spread = _spread(swarm.positions())

    for step in range(steps):
        swarm.step(dt)
        if step % every == 0 or step == steps - 1:
            pos = swarm.positions()
            vel = swarm.velocities()
            sp  = _spread(pos)
            spd = _avg_speed(vel)
            label = _flock_label(sp, spd, initial_spread)
            t = (step + 1) * dt
            print(f"{step+1:>5}  {t:>5.2f}s  {sp:>7.3f}m  {spd:>8.3f}m/s  {label}")

    print()
    final_spread = _spread(swarm.positions())
    change_pct = (initial_spread - final_spread) / initial_spread * 100
    print(f"Initial spread: {initial_spread:.3f}m")
    print(f"Final spread:   {final_spread:.3f}m  "
          f"({'tighter' if change_pct > 0 else 'wider'} by {abs(change_pct):.1f}%)")
    if change_pct > 5:
        print("→ Cohesion rule worked: the flock pulled together over time.")
    return 0


def _cmd_check(args) -> int:
    path = pathlib.Path(args.file)
    try:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source, filename=str(path))
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except SyntaxError as exc:
        print(f"{path}:{exc.lineno}: syntax error: {exc.msg}", file=sys.stderr)
        return 1

    source_lines  = source.splitlines()
    error_count   = 0
    warning_count = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id    if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if name != "Swarm" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, int)):
            continue

        n_boids   = first.value
        col       = node.col_offset + 1
        src_line  = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
        token_len = len(f"Swarm({n_boids})")

        for diag in _diagnose(n_boids):
            print(_fmt_check(diag, str(path), node.lineno, col, src_line, token_len))
            if diag.level == _Level.ERROR:
                error_count += 1
            elif diag.level == _Level.WARNING:
                warning_count += 1

    total = error_count + warning_count
    if total:
        parts = []
        if error_count:
            parts.append(f"{error_count} error(s)")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")
        print(f"{path}: {', '.join(parts)}")
        return 2 if args.error else 1

    print(f"{path}: ok")
    return 0


def _cmd_info() -> int:
    import platform
    from . import __version__

    print(f"zombiebubble {__version__}")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
