"""
python/zombiebubble/cli.py

Entry point for the `zombiebubble` command-line tool.
"""

from __future__ import annotations

import argparse
import math
import sys

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

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args)
    elif args.cmd == "info":
        return _cmd_info()
    else:
        parser.print_help()
        return 0


def _cmd_run(args) -> int:
    from .simulation import Swarm

    n, steps, dt, every = args.boids, args.steps, args.dt, args.print_every

    if n < 1 or n > 4096:
        print("error: --boids must be 1–4096", file=sys.stderr)
        return 1

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


def _cmd_info() -> int:
    import platform
    from . import __version__

    print(f"zombiebubble {__version__}")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
