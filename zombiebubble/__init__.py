"""
zombiebubble – ECS boids simulation (free tier)

Quick start:
    from zombiebubble import Swarm

    s = Swarm(32)
    for _ in range(200):
        s.step(0.016)
    print(s.positions())
"""

from .simulation import Swarm

__all__ = ["Swarm"]
__version__ = "0.1.0"
