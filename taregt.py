import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# # Cube dimensions
# length = 3.5  # meters
# width = 3.5  # meters
# height = 2.0  # meters

# # Grid spacing
# grid_spacing = 0.2  # meters

# # Generate grid points
# x = np.arange(0, length + grid_spacing, grid_spacing)
# y = np.arange(0, width + grid_spacing, grid_spacing)
# z = np.arange(0, height + grid_spacing, grid_spacing)

# # Create a figure with higher resolution
# fig = plt.figure(figsize=(10, 10), dpi=300)
# ax = fig.add_subplot(111, projection='3d')

# # Set a perspective view
# ax.view_init(elev=25, azim=30)  # Adjust elevation and azimuth for perspective

# # Draw grid inside the cube
# for xi in x:
#     for yi in y:
#         ax.plot([xi, xi], [yi, yi], [0, height], color='gray', linewidth=0.5)  # Vertical lines

# for xi in x:
#     for zi in z:
#         ax.plot([xi, xi], [0, width], [zi, zi], color='gray', linewidth=0.5)  # XZ plane lines

# for yi in y:
#     for zi in z:
#         ax.plot([0, length], [yi, yi], [zi, zi], color='gray', linewidth=0.5)  # YZ plane lines

# # Draw the outer cube with perspective
# cube_vertices = [
#     [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0],  # Bottom face
#     [0, 0, height], [length, 0, height], [length, width, height], [0, width, height]  # Top face
# ]
# edges = [
#     (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
#     (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
#     (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
# ]
# for edge in edges:
#     ax.plot(*zip(*[cube_vertices[i] for i in edge]), color='black', linewidth=1.5)

# # Labels and limits
# ax.set_xlabel("X Axis (m)")
# ax.set_ylabel("Y Axis (m)")
# ax.set_zlabel("Z Axis (m)")
# ax.set_xlim(0, length)
# ax.set_ylim(0, width)
# ax.set_zlim(0, height)

# plt.title("3D Cube with Perspective and 20cm Grid")
# plt.show()



# Re-import necessary libraries after execution state reset
# Re-import necessary libraries after execution state reset

# Camera parameters
height = 10  # Height of the pyramid (camera FOV depth)
fov_h = 40  # Horizontal field of view (degrees)
fov_v = 55  # Vertical field of view (degrees)

# Convert FOV to radians
fov_h_rad = np.radians(fov_h / 2)
fov_v_rad = np.radians(fov_v / 2)

# Compute base dimensions of the pyramid
base_half_width = np.tan(fov_h_rad) * height
base_half_height = np.tan(fov_v_rad) * height

# Define pyramid vertices (Apex at origin)
apex = np.array([0, 0, 0])  # Camera position
v1 = np.array([base_half_width, base_half_height, -height])
v2 = np.array([-base_half_width, base_half_height, -height])
v3 = np.array([-base_half_width, -base_half_height, -height])
v4 = np.array([base_half_width, -base_half_height, -height])

# Create figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Define pyramid faces
faces = [
    [apex, v1, v2],  # Front face
    [apex, v2, v3],  # Left face
    [apex, v3, v4],  # Back face
    [apex, v4, v1],  # Right face
    [v1, v2, v3, v4]  # Base
]

# Add the pyramid to the plot
pyramid = Poly3DCollection(faces, alpha=0.2, edgecolor="black")
ax.add_collection3d(pyramid)

# Generate rays from the apex to the base
num_rays = 50  # Number of rays inside the pyramid
for _ in range(num_rays):
    # Random point inside the base plane
    rand_x = np.random.uniform(-base_half_width, base_half_width)
    rand_y = np.random.uniform(-base_half_height, base_half_height)
    rand_z = -height  # All rays end at the base
    ax.plot([apex[0], rand_x], [apex[1], rand_y], [apex[2], rand_z], color="red", linewidth=0.5)

# Labels and limits
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_xlim(-base_half_width, base_half_width)
ax.set_ylim(-base_half_height, base_half_height)
ax.set_zlim(-height, 1)

plt.title("3D Camera FOV Pyramid with Rays")
plt.show()
