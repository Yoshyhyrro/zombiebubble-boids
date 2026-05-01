"""
python/tests/test_flocking.py

Behaviour-level tests for the boids simulation.

Each test name and docstring explains *what is happening physically*,
so they serve as documentation for people unfamiliar with Boids.

Run with:  cd python && pytest -v
"""
import math
import pytest
from zombiebubble import Swarm
from zombiebubble._core import vec3_dot, vec3_normalize, vec3_length, vec3_cross


# ── helpers ────────────────────────────────────────────────────────────────────

def spread(positions):
    """Average distance from centroid — smaller = tighter flock."""
    n = len(positions)
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n
    cz = sum(p[2] for p in positions) / n
    return sum(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
               for p in positions) / n


def avg_speed(velocities):
    return sum(math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in velocities) / len(velocities)


# ── Flocking behaviour tests ───────────────────────────────────────────────────

def test_cohesion_pulls_flock_together():
    """
    COHESION rule: each boid steers toward the average position of its
    neighbours — like a school of fish staying together.

    Observable effect: the average distance between agents (spread) should
    decrease over time as the flock clusters.
    """
    swarm = Swarm(20)
    initial = spread(swarm.positions())

    for _ in range(300):   # ~5 simulated seconds
        swarm.step(0.016)

    final = spread(swarm.positions())
    assert final < initial, (
        f"Cohesion should pull the flock together.\n"
        f"  initial spread = {initial:.3f}m\n"
        f"  final spread   = {final:.3f}m\n"
        f"  (final should be smaller)"
    )


def test_separation_prevents_boids_from_stacking():
    """
    SEPARATION rule: boids steer away from neighbours that are too close —
    like people in a crowd keeping personal space.

    Observable effect: after the simulation settles, no two boids should
    occupy the same position (minimum pairwise distance > threshold).
    """
    swarm = Swarm(16)
    for _ in range(200):
        swarm.step(0.016)

    pos = swarm.positions()
    n = len(pos)
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dz = pos[i][2] - pos[j][2]
            min_dist = min(min_dist, math.sqrt(dx*dx + dy*dy + dz*dz))

    assert min_dist > 0.01, (
        f"Separation should prevent boids from overlapping.\n"
        f"  minimum distance between any two boids = {min_dist:.4f}m\n"
        f"  (should be > 0.01m)"
    )


def test_alignment_keeps_flock_moving():
    """
    ALIGNMENT rule: boids match the velocity of their neighbours —
    like geese flying in formation.

    Observable effect: once aligned, the whole flock keeps moving together
    rather than coming to a stop.  Average speed should remain above zero.
    """
    swarm = Swarm(20)
    for _ in range(200):
        swarm.step(0.016)

    spd = avg_speed(swarm.velocities())
    assert spd > 0.01, (
        f"Boids should keep moving after the flock aligns.\n"
        f"  average speed = {spd:.4f} m/s  (should be > 0.01)"
    )


def test_flock_centroid_stays_bounded():
    """
    The three forces (cohesion, alignment, separation) balance each other.
    The flock's centre of mass should not fly off to infinity.
    """
    swarm = Swarm(20)
    for _ in range(300):
        swarm.step(0.016)

    pos = swarm.positions()
    n = len(pos)
    cx = sum(p[0] for p in pos) / n
    cy = sum(p[1] for p in pos) / n
    cz = sum(p[2] for p in pos) / n
    dist = math.sqrt(cx**2 + cy**2 + cz**2)

    assert dist < 30.0, (
        f"Flock centroid drifted too far from origin.\n"
        f"  centroid = ({cx:.2f}, {cy:.2f}, {cz:.2f})\n"
        f"  distance from origin = {dist:.2f}m  (should be < 30m)"
    )


# ── Vec3 math tests (with physical meaning) ───────────────────────────────────

def test_dot_product_detects_facing_direction():
    """
    The dot product of two direction vectors tells us whether they point
    in the same (>0), opposite (<0), or perpendicular (≈0) directions.

    In boids: used to check whether a neighbour is ahead or behind
    a boid (field-of-view cone).
    """
    forward  = (1.0, 0.0, 0.0)
    same     = (0.8, 0.0, 0.6)   # roughly same direction
    opposite = (-1.0, 0.0, 0.0)
    sideways = (0.0,  1.0, 0.0)

    assert vec3_dot(forward, same)     > 0,      "neighbour ahead → dot > 0"
    assert vec3_dot(forward, opposite) < 0,      "neighbour behind → dot < 0"
    assert abs(vec3_dot(forward, sideways)) < 1e-6, "neighbour sideways → dot ≈ 0"


def test_normalize_gives_unit_direction_vector():
    """
    Normalizing a vector strips its magnitude and keeps only its direction,
    giving a unit vector (length = 1).

    In boids: force contributions are normalized before being scaled by
    their weights, so distance doesn't distort the force balance.
    """
    v = (3.0, 4.0, 0.0)   # length = 5
    nx, ny, nz = vec3_normalize(v)
    length = vec3_length((nx, ny, nz))

    assert abs(length - 1.0) < 1e-6, (
        f"Normalized vector should have length 1, got {length:.6f}"
    )
    assert abs(nx - 0.6) < 1e-5, f"x component: expected 0.6, got {nx}"
    assert abs(ny - 0.8) < 1e-5, f"y component: expected 0.8, got {ny}"


def test_cross_product_gives_perpendicular_vector():
    """
    The cross product of two vectors produces a third vector perpendicular
    to both.

    In boids: used for vortex force directions and torque computation.
    """
    a = (1.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0)
    cx, cy, cz = vec3_cross(a, b)

    # a × b should point along +Z
    assert abs(cx) < 1e-6,       f"x should be 0, got {cx}"
    assert abs(cy) < 1e-6,       f"y should be 0, got {cy}"
    assert abs(cz - 1.0) < 1e-6, f"z should be 1, got {cz}"

    # Result is perpendicular to both inputs
    assert abs(vec3_dot(a, (cx, cy, cz))) < 1e-6, "result should be ⊥ to a"
    assert abs(vec3_dot(b, (cx, cy, cz))) < 1e-6, "result should be ⊥ to b"
