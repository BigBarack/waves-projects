import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation

# Define the domain
outer_boundary = 1.0  # Length of the outer square domain
inner_boundary = 0.4  # Length of the inner square boundary (Gamma 1)
num_points_outer = 50  # Points per side on the outer square
num_points_inner = 20  # Points per side on the inner square

# Generate vertices for the outer boundary (Gamma 2)
outer_vertices = []
for i in range(num_points_outer):
    t = i / (num_points_outer - 1)
    outer_vertices += [
        [t * outer_boundary, 0],
        [outer_boundary, t * outer_boundary],
        [outer_boundary * (1 - t), outer_boundary],
        [0, outer_boundary * (1 - t)],
    ]
outer_vertices = np.unique(outer_vertices, axis=0)

# Generate vertices for the inner boundary (Gamma 1)
inner_vertices = []
for i in range(num_points_inner):
    t = i / (num_points_inner - 1)
    inner_vertices += [
        [t * inner_boundary + (outer_boundary - inner_boundary) / 2, (outer_boundary - inner_boundary) / 2],
        [(outer_boundary + inner_boundary) / 2, t * inner_boundary + (outer_boundary - inner_boundary) / 2],
        [outer_boundary - t * inner_boundary - (outer_boundary - inner_boundary) / 2, (outer_boundary + inner_boundary) / 2],
        [(outer_boundary - inner_boundary) / 2, outer_boundary - t * inner_boundary - (outer_boundary - inner_boundary) / 2],
    ]
inner_vertices = np.unique(inner_vertices, axis=0)

# Combine inner and outer vertices
vertices = np.vstack([outer_vertices, inner_vertices])

# Generate triangles (simple Delaunay triangulation for demonstration)
from scipy.spatial import Delaunay
tri = Delaunay(vertices)
triangles = tri.simplices

# Generate fake wave data for demonstration
time_steps = 100
wave_data = []
for t in range(time_steps):
    u_t = np.sin(2 * np.pi * t / time_steps) * (np.linalg.norm(vertices, axis=1) + 0.5)
    wave_data.append(u_t)

wave_data = np.array(wave_data)

# Setup the triangulation
triangulation = Triangulation(vertices[:, 0], vertices[:, 1], triangles)

# Animation function
def animate(frame):
    z = wave_data[frame]
    plot.set_array(z)
    return plot,

# Plot initial frame
fig, ax = plt.subplots()
plot = ax.tripcolor(triangulation, wave_data[0], shading='flat', cmap='viridis')
ax.set_aspect('equal')
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
plt.colorbar(plot)

# Create animation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=True)
plt.show()