# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Camera parameters
height = 30  # Height of the pyramid (camera FOV depth)
fov_h = 20  # Horizontal field of view (degrees)
fov_v = 25  # Vertical field of view (degrees)

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
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')

# Define pyramid faces
faces = [
    [apex, v1, v2],  # Front face
    [apex, v2, v3],  # Left face
    # [apex, v3, v4],  # Back face
    [apex, v4, v1],  # Right face
    [v1, v2, v3, v4]  # Base
]

# Add the pyramid to the plot
pyramid = Poly3DCollection(faces, alpha=0.08, edgecolor="black")
ax.add_collection3d(pyramid)

# Generate rays from the apex to the base
num_rays = 1  # Number of rays inside the pyramid
for _ in range(num_rays):
    # Random point inside the base plane
    rand_x = np.random.uniform(-base_half_width, base_half_width)
    rand_y = np.random.uniform(-base_half_height, base_half_height)
    rand_z = -height  # All rays end at the base
    ax.plot([apex[0], rand_x], [apex[1], rand_y], [apex[2], rand_z], color="pink", linewidth=0.7)

# Labels and limits
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_xlim(-base_half_width, base_half_width)
ax.set_ylim(-base_half_height, base_half_height)
ax.set_zlim(-height, 1)

plt.title("3D Camera FOV Pyramid with Rays")
ax.grid(False)

plt.show()
