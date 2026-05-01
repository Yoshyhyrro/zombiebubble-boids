import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from zombiebubble import Swarm

def main():
    print('Generating boids.gif...')
    num_boids = 150
    steps = 250
    dt = 0.016
    swarm = Swarm(num_boids)
    
    history = []
    for _ in range(steps):
        swarm.step(dt)
        history.append(np.array(swarm.positions()))
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Boidswarm (Graph-Laplacian ECS)')
    ax.set_axis_off()
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    scatter = ax.scatter(history[0][:, 0], history[0][:, 1], history[0][:, 2], s=15, c='dodgerblue', marker='^')
    
    def update(frame):
        scatter._offsets3d = (history[frame][:, 0], history[frame][:, 1], history[frame][:, 2])
        # Slowly rotate the view
        ax.view_init(elev=20., azim=frame * 0.5)
        return scatter,
        
    anim = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
    anim.save('boids.gif', writer='pillow', fps=50)
    print('Done! Saved to boids.gif')

if __name__ == '__main__':
    main()
