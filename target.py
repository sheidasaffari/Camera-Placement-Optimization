import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cube dimensions
length = 3.5  # meters
width = 3.5  # meters
height = 2.0  # meters

# Grid spacing
grid_spacing = 0.5  # meters

# Generate grid points
x = np.arange(0, length + grid_spacing, grid_spacing)
y = np.arange(0, width + grid_spacing, grid_spacing)
z = np.arange(0, height + grid_spacing, grid_spacing)

# Create a figure
fig = plt.figure(figsize=(8, 8), dpi = 300)
ax = fig.add_subplot(111, projection='3d')

# Draw grid inside the cube
for xi in x:
    for yi in y:
        ax.plot([xi, xi], [yi, yi], [0, height], color='gray', linewidth=0.5)  # Vertical lines

for xi in x:
    for zi in z:
        ax.plot([xi, xi], [0, width], [zi, zi], color='gray', linewidth=0.5)  # XZ plane lines

for yi in y:
    for zi in z:
        ax.plot([0, length], [yi, yi], [zi, zi], color='gray', linewidth=0.5)  # YZ plane lines

# Draw the outer cube
cube_vertices = [
    [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0],  # Bottom face
    [0, 0, height], [length, 0, height], [length, width, height], [0, width, height]  # Top face
]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
]
for edge in edges:
    ax.plot(*zip(*[cube_vertices[i] for i in edge]), color='black', linewidth=1.5)

# Labels and limits
ax.set_xlabel("X Axis (m)")
ax.set_ylabel("Y Axis (m)")
ax.set_zlabel("Z Axis (m)")
ax.set_xlim(0, length)
ax.set_ylim(0, width)
ax.set_zlim(0, height)

plt.title("3D Cube with 20cm Grid")
plt.show()
