import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from itertools import cycle
from datetime import datetime
import torch
import pyswarms as ps
from scipy.special import comb

# --------------------------
# User-Defined Hyper-Parameters
# --------------------------
# Room and target box dimensions
room_size = {'width': 5.181, 'length': 5.308, 'height': 2.700}
box_size = {'width': 3.581, 'length': 3.708, 'height': 0.700}
box_position = {'x': (room_size['width'] - box_size['width']) / 2,
                'y': (room_size['length'] - box_size['length']) / 2,
                'z': 0}

# Compute box center and room center (for later use)
box_center = {'x': box_position['x'] + box_size['width']/2,
              'y': box_position['y'] + box_size['length']/2,
              'z': box_position['z'] + box_size['height']/2}
room_width = room_size['width']
room_length = room_size['length']
room_height = room_size['height']
center_of_floor = np.array([room_width/2, room_length/2, 0])

# Camera parameters
camera_fov = {'horizontal': 40, 'vertical': 60}
max_depth = 4       # Maximum FOV depth
voxel_size = 0.4    # Voxel resolution for target space
num_params_per_camera = 6  # [x, y, z, pitch, yaw, roll]
num_cameras = 3

# Objective function weights (here only coverage is used, but could be modified)
W1 = 0.7      # Weight for coverage
W2 = 0.15   # Weight for scale differences 
W3 = 0.15      # Weight for viewpoint diversity 

# --------------------------
# Geometry & Search Space Helpers
# --------------------------
def is_valid_position(x, y, z, room_dim=room_size, box_cord=box_position, box_dim=box_size):
    """
    Returns True if (x,y,z) lies within the room but outside the box.
    """
    pt = Point(x, y)
    room_poly = box(0, 0, room_dim['width'], room_dim['length'])
    box_poly = box(box_cord['x'], box_cord['y'], 
                   box_cord['x']+box_dim['width'], 
                   box_cord['y']+box_dim['length'])
    # For z in the range of the box, the (x,y) must be outside the box polygon.
    if box_cord['z'] <= z <= (box_cord['z']+box_dim['height']):
        return room_poly.difference(box_poly).contains(pt)
    # For z above the box, entire room is valid.
    elif z > (box_cord['z']+box_dim['height']) and room_poly.contains(pt):
        return True
    else:
        return False

def repair_position(pos):
    """
    If a camera position pos = [x, y, z] lies inside the box,
    repair it by moving it to the nearest boundary.
    This function assumes the box is centrally located.
    """
    x, y, z = pos
    if is_valid_position(x, y, z):
        return np.array(pos)
    # Otherwise, repair the (x,y) coordinates.
    # For simplicity, if (x,y) is inside the box and z is in [box_z, box_z+box_h],
    # we move the point to the closest edge.
    left = box_position['x']
    right = box_position['x'] + box_size['width']
    bottom = box_position['y']
    top = box_position['y'] + box_size['length']
    # Compute distances to each side in x and y
    dx_left = abs(x - left)
    dx_right = abs(x - right)
    dy_bottom = abs(y - bottom)
    dy_top = abs(y - top)
    # Adjust the coordinate that is closest to the boundary
    if dx_left < dx_right and dx_left <= dy_bottom and dx_left <= dy_top:
        x = left
    elif dx_right < dx_left and dx_right <= dy_bottom and dx_right <= dy_top:
        x = right
    elif dy_bottom < dy_top:
        y = bottom
    else:
        y = top
    # Also, if z is within the box height, set it to box top.
    if box_position['z'] <= z <= box_position['z']+box_size['height']:
        z = box_position['z'] + box_size['height']
    return np.array([x, y, z])

def calculate_fov_pyramid(camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll):
    """
    Computes the 4 corners of the base of the FOV pyramid.
    Applies rotations in the order roll, then pitch, then yaw.
    """
    half_h = math.radians(h_fov/2)
    half_v = math.radians(v_fov/2)
    dx = max_depth * math.tan(half_h)
    dy = max_depth * math.tan(half_v)
    # Define the corners relative to the camera (before rotation)
    corners = np.array([
        [camera_pos[0]-dx, camera_pos[1]-dy, camera_pos[2]-max_depth],
        [camera_pos[0]+dx, camera_pos[1]-dy, camera_pos[2]-max_depth],
        [camera_pos[0]+dx, camera_pos[1]+dy, camera_pos[2]-max_depth],
        [camera_pos[0]-dx, camera_pos[1]+dy, camera_pos[2]-max_depth]
    ])
    # Build rotation matrices
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    R_yaw = np.array([[math.cos(yaw_rad), 0, math.sin(yaw_rad)],
                      [0, 1, 0],
                      [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                        [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    R_roll = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0],
                       [math.sin(roll_rad), math.cos(roll_rad), 0],
                       [0, 0, 1]])
    # Apply rotations about the camera position
    rotated = []
    for corner in corners:
        v = corner - np.array(camera_pos)
        v_rot = R_yaw.dot(R_pitch.dot(R_roll.dot(v)))
        rotated.append((np.array(camera_pos) + v_rot).tolist())
    return rotated

def voxel_positions():
    """
    Generate voxel center positions within the target box.
    """
    voxels = []
    for x in np.arange(box_position['x'], box_position['x']+box_size['width'], voxel_size):
        for y in np.arange(box_position['y'], box_position['y']+box_size['length'], voxel_size):
            for z in np.arange(box_position['z'], box_position['z']+box_size['height'], voxel_size):
                voxels.append([x+voxel_size/2, y+voxel_size/2, z+voxel_size/2])
    return voxels

# Precompute voxel centers for the target box.
voxel_coord = voxel_positions()

def Is_inside_pyramid(voxel, camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll):
    """
    Determine if a given voxel (point) lies within the FOV pyramid.
    """
    floor_corners = calculate_fov_pyramid(camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll)
    center_floor = np.mean(np.array(floor_corners), axis=0)
    optical_line = center_floor - np.array(camera_pos)
    norm_optical = np.linalg.norm(optical_line)
    if norm_optical == 0:
        return False
    optical_unit = optical_line / norm_optical
    point_vec = np.array(voxel) - np.array(camera_pos)
    proj_length = np.dot(point_vec, optical_unit)
    if proj_length < 0 or proj_length > max_depth:
        return False
    closest_point = np.array(camera_pos) + proj_length * optical_unit
    perp_dist = np.linalg.norm(np.array(voxel) - closest_point)
    # For the given depth, compute the effective half-widths:
    curr_dx = (proj_length * math.tan(math.radians(h_fov/2)))
    curr_dy = (proj_length * math.tan(math.radians(v_fov/2)))
    # Use an approximate ellipse test:
    if (perp_dist <= np.sqrt(curr_dx**2 + curr_dy**2)):
        return True
    return False

def camera_voxel_matrix(camera_poses, pitches, yaws, rolls, points):
    """
    Build a binary matrix where each element is 1 if the voxel is inside the camera's FOV.
    """
    M = np.zeros((len(camera_poses), len(points)))
    for i, cam in enumerate(camera_poses):
        for j, pt in enumerate(points):
            if Is_inside_pyramid(pt, cam, camera_fov['horizontal'], camera_fov['vertical'], max_depth, pitches[i], yaws[i], rolls[i]):
                M[i, j] = 1
    return M

def camera_distance_differences(camera_poses, points):
    """
    Compute pairwise absolute differences in distances from each camera pair to each voxel.
    """
    num_pairs = len(camera_poses)*(len(camera_poses)-1)//2
    diff_matrix = np.zeros((num_pairs, len(points)))
    idx = 0
    for i in range(len(camera_poses)):
        for j in range(i+1, len(camera_poses)):
            for k, pt in enumerate(points):
                d_i = np.linalg.norm(np.array(pt) - np.array(camera_poses[i]))
                d_j = np.linalg.norm(np.array(pt) - np.array(camera_poses[j]))
                diff_matrix[idx, k] = abs(d_i - d_j)
            idx += 1
    return diff_matrix

def camera_angles(camera_poses, points):
    """
    Compute pairwise angles between vectors from cameras to each voxel.
    """
    angles = []
    for i in range(len(camera_poses)):
        for j in range(i+1, len(camera_poses)):
            pair_angles = []
            for pt in points:
                v1 = np.array(pt) - np.array(camera_poses[i])
                v2 = np.array(pt) - np.array(camera_poses[j])
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = math.degrees(math.acos(cos_angle))
                pair_angles.append(angle_deg)
            angles.append(pair_angles)
    return np.array(angles)

# --------------------------
# Orientation and Initial Positions
# --------------------------
def calculate_orientation(camera_pos, target_pos):
    """
    Calculate orientation (pitch, yaw, roll) for a camera to look toward a target.
    For our purposes, roll is kept at 0.
    """
    vector = np.array(target_pos) - np.array(camera_pos)
    dx, dy, dz = vector
    yaw = math.degrees(math.atan2(dx, dz)) + 180
    dist_xz = math.sqrt(dx**2 + dz**2)
    pitch = -math.degrees(math.atan2(-dy, dist_xz))
    roll = 0
    # Normalize angles to [-180,180]
    yaw = ((yaw + 180) % 360) - 180
    pitch = ((pitch + 180) % 360) - 180
    return pitch, yaw, roll

def create_initial_positions(num_cameras, num_particles, min_bounds, max_bounds):
    """
    Generate only valid initial positions (outside the box) for all cameras.
    """
    init_pos = np.zeros((num_particles, num_cameras * num_params_per_camera))
    for i in range(num_particles):
        for j in range(num_cameras):
            idx = j * num_params_per_camera
            valid = False
            while not valid:
                x = np.random.uniform(min_bounds[idx], max_bounds[idx])
                y = np.random.uniform(min_bounds[idx+1], max_bounds[idx+1])
                z = np.random.uniform(min_bounds[idx+2], max_bounds[idx+2])
                if is_valid_position(x, y, z):
                    valid = True
            camera_pos = np.array([x, y, z])
            target_pos = [box_center['x'], box_center['y'], 0]
            pitch, yaw, roll = calculate_orientation(camera_pos, target_pos)
            init_pos[i, idx:idx+num_params_per_camera] = [x, y, z, pitch, yaw, roll]
    return init_pos

# --------------------------
# PSO Fitness Function with Repair Operator
# --------------------------
def pso_fitness_function_only_coverage(particles):
    """
    Evaluate each particle by repairing any invalid camera positions and computing the cost.
    """
    fitness_scores = np.zeros(particles.shape[0])
    for i, flat_params in enumerate(particles):
        # Reshape flat parameters to (num_cameras, 6)
        params = flat_params.reshape((num_cameras, num_params_per_camera))
        # Repair camera positions if they fall inside the box
        for idx in range(num_cameras):
            params[idx, :3] = repair_position(params[idx, :3])
        camera_poses = params[:, :3]
        pitches = params[:, 3]
        yaws = params[:, 4]
        rolls = params[:, 5]
        # Compute the voxel coverage matrix and sum over voxels
        big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)
        sum_columns = np.clip(np.sum(big_matrix, axis=0), 0, 1)
        num_voxels_covered = np.sum(sum_columns)
        normalized_coverage = num_voxels_covered / len(voxel_coord)

        # (Other terms can be added here if needed)
        cost = W1 * normalized_coverage
        # We return the negative cost as PSO minimizes the function.
        fitness_scores[i] = -cost
    return fitness_scores


def pso_fitness_function(particles):
    """
    Evaluate each particle by first repairing any invalid camera positions
    (positions inside the target box) and then computing three metrics:
      1. Normalized coverage (voxel coverage)
      2. Normalized pairwise distance differences
      3. Normalized pairwise angles between cameras
      
    The final cost is given by:
      cost = W1 * normalized_coverage + W2 * normalized_distance_differences + W3 * normalized_angles
      
    Returns an array of fitness scores (to be minimized) for each particle.
    """
    fitness_scores = np.zeros(particles.shape[0])
    
    for i, flat_params in enumerate(particles):
        # Reshape the flat parameter vector into (num_cameras, num_params_per_camera)
        params = flat_params.reshape((num_cameras, num_params_per_camera))
        # Repair each camera's (x,y,z) if it falls inside the target box
        for idx in range(num_cameras):
            params[idx, :3] = repair_position(params[idx, :3])
        camera_poses = params[:, :3]
        pitches = params[:, 3]
        yaws = params[:, 4]
        rolls = params[:, 5]
        
        # Compute voxel coverage
        big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)
        sum_columns = np.clip(np.sum(big_matrix, axis=0), 0, 1)
        num_voxels_covered = np.sum(sum_columns)
        normalized_coverage = num_voxels_covered / len(voxel_coord)
        
        # Compute pairwise distance differences (sum over all voxel evaluations)
        distance_differences = camera_distance_differences(camera_poses, voxel_coord)
        sum_distance_differences = np.sum(distance_differences)
        # Normalize by the maximum possible distance (diagonal of room) times number of voxel evaluations and number of pairs
        normalization_factor = comb(num_cameras, 2) * len(voxel_coord) * np.sqrt(room_size['width']**2 + room_size['length']**2 + room_size['height']**2)
        normalized_distance_differences = sum_distance_differences / normalization_factor
        
        # Compute pairwise angles
        angles = camera_angles(camera_poses, voxel_coord)
        sum_angles = np.sum(angles)
        # Normalize by 180° per voxel evaluation per camera pair
        normalized_angles = sum_angles / (comb(num_cameras, 2) * len(voxel_coord) * 180)
        
        # Combine the objectives with the respective weights
        normalized_fitness_score = W1 * normalized_coverage + W2 * normalized_distance_differences + W3 * normalized_angles
        
        # Since PSO minimizes the fitness value, return its negative.
        fitness_scores[i] = -normalized_fitness_score

    return fitness_scores

# --------------------------
# Visualization Function
# --------------------------
def visualize_cameras_and_fov(cameras_solution):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    # Draw room
    room_corners = [[0,0,0],
                    [room_size['width'], 0, 0],
                    [room_size['width'], room_size['length'], 0],
                    [0, room_size['length'], 0],
                    [0,0,room_size['height']],
                    [room_size['width'],0,room_size['height']],
                    [room_size['width'],room_size['length'],room_size['height']],
                    [0,room_size['length'],room_size['height']]]
    room_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in room_edges:
        ax.plot3D(*zip(room_corners[e[0]], room_corners[e[1]]), color='cyan')
    # Draw box (target) in green
    bmin = [box_position['x'], box_position['y'], box_position['z']]
    bmax = [box_position['x']+box_size['width'], box_position['y']+box_size['length'], box_position['z']+box_size['height']]
    box_corners = [[bmin[0],bmin[1],bmin[2]],
                   [bmax[0],bmin[1],bmin[2]],
                   [bmax[0],bmax[1],bmin[2]],
                   [bmin[0],bmax[1],bmin[2]],
                   [bmin[0],bmin[1],bmax[2]],
                   [bmax[0],bmin[1],bmax[2]],
                   [bmax[0],bmax[1],bmax[2]],
                   [bmin[0],bmax[1],bmax[2]]]
    box_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in box_edges:
        ax.plot3D(*zip(box_corners[e[0]], box_corners[e[1]]), color='green')
    # Draw cameras and their FOVs
    colors = cycle(['red','blue','magenta','orange'])
    for cam, col in zip(cameras_solution, colors):
        pos = cam[:3]
        pitch, yaw, roll = cam[3], cam[4], cam[5]
        fov_pts = calculate_fov_pyramid(pos, camera_fov['horizontal'], camera_fov['vertical'], max_depth, pitch, yaw, roll)
        for pt in fov_pts:
            ax.plot([pos[0], pt[0]], [pos[1], pt[1]], [pos[2], pt[2]], color=col)
        ax.scatter(*pos, color=col, s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Optimized Camera Positions and FOVs')
    plt.show()

# --------------------------
# PSO Optimization Setup
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# np.random.seed(18)
maxiter = 500
num_particles = 50
# iteration = 200
# PSO dimensions: 6 parameters per camera
dimensions = num_cameras * num_params_per_camera

# For PSO bounds, we use the room bounds for x, y, z and angle bounds for orientation.
# Note: The valid search space is the room minus the box.
param_bounds = [
    (0, room_size['width']),      # x
    (0, room_size['length']),     # y
    (0, room_size['height']),     # z
    (-180, 180),                  # pitch
    (-180, 180),                  # yaw
    (-1, 1)                       # roll (if roll is fixed, these bounds are narrow)
]
# Expand bounds for all cameras
min_bounds = np.array([b[0] for b in param_bounds] * num_cameras)
max_bounds = np.array([b[1] for b in param_bounds] * num_cameras)

# Create initial positions from valid search space
init_pos = create_initial_positions(num_cameras, num_particles, min_bounds, max_bounds)

# Convert initial positions to tensor if needed (PSO library expects NumPy)
init_pos_np = init_pos

# PSO options (tune as needed)
options = {'c1': 0.7, 'c2': 1.2, 'w': 0.7}

optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                    dimensions=dimensions,
                                    options=options,
                                    bounds=(min_bounds, max_bounds),
                                    init_pos=init_pos_np)

# Perform optimization
# cost, pos = optimizer.optimize(pso_fitness_function, iters=iteration)
# optimal_camera_poses = pos.reshape((-1, num_params_per_camera))
# print("Optimized Camera Poses:", optimal_camera_poses)

# Extract camera parameters
# camera_poses = optimal_camera_poses[:, :3]
# pitches = optimal_camera_poses[:, 3]
# yaws = optimal_camera_poses[:, 4]
# rolls = optimal_camera_poses[:, 5]

# # Evaluate coverage and pairwise metrics
# big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)
# camera_coverage_counts = np.sum(big_matrix, axis=1)
# total_voxels = len(voxel_coord)
# coverage_percentages = (camera_coverage_counts / total_voxels) * 100
# num_voxels_covered = np.sum(np.clip(np.sum(big_matrix, axis=0), 0, 1))

# # Compute pairwise metrics for reporting
# distance_diffs = camera_distance_differences(camera_poses, voxel_coord)
# angles = camera_angles(camera_poses, voxel_coord)
# avg_distance_diff = np.mean(distance_diffs)
# avg_angle = np.mean(angles)

# print('Camera Coverage Counts:', camera_coverage_counts)
# print('Coverage Percentages:', coverage_percentages)
# print('Total Voxels Covered:', num_voxels_covered)
# print('Average Pairwise Distance Difference:', avg_distance_diff)
# print('Average Pairwise Angle (deg):', avg_angle)


### Plot the cost history

# cost_history = optimizer.cost_history

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(len(cost_history)), -np.array(cost_history), marker='o', linestyle='-')
# plt.xlabel('Iteration')
# plt.ylabel('Fitness Value (-cost)')
# plt.title('PSO Cost History')
# plt.grid(True)
# plt.show()

######################Start##############################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################

# To run this part of the code, above lines where we are doing
# the optimization and calculating teh cost and best pose should be commented
# For the purpose of computational resources, we used only 5 values in teh space, 
# then we will limit this bound and find more optimum hyperparameetrs

iteration = 20
# Set the seed for reproducibility
np.random.seed(220)

# Define the parameter grid
w_values = np.linspace(0.2, 1, num = 10)
c1_values = np.linspace(0.5, 2, num= 10)
c2_values = np.linspace(0.5, 2, num= 10)

results  = []



for c1 in c1_values:
    for c2 in c2_values:
        for w in w_values:
            # Update the options with the current set of parameters
            options = {'c1': c1, 'c2': c2, 'w': w}

            # Initialize the optimizer with the current set of parameters
            optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                                dimensions=dimensions,
                                                options=options,
                                                bounds=(min_bounds, max_bounds),
                                                init_pos=init_pos)
            
            # Perform optimization
            cost, pos = optimizer.optimize(pso_fitness_function, iters=iteration)
            
            # Store the results along with the corresponding parameters
            results.append((cost, c1, c2, w, num_cameras, W1, W2, W3, iteration))


# Convert results to a DataFrame for easier analysis
df_results = pd.DataFrame(results, columns=['best_cost', 'c1', 'c2', 'w', 'num_cameras', 'W1', 'W2', 'W3', 'iteration'])



# Example: Print the top 5 parameter combinations with the lowest cost
print(df_results.sort_values(by='best_cost').head())

# You could also create plots to visualize the sensitivity
# For instance, plotting the best cost for different values of w
plt.figure(figsize=(10, 6))
for c1 in c1_values:
    for c2 in c2_values:
        subset = df_results[(df_results['c1'] == c1) & (df_results['c2'] == c2)]
        plt.plot(subset['w'], subset['best_cost'], marker='o', label=f'c1={c1}, c2={c2}')

plt.xlabel('Inertia Weight (w)')
plt.ylabel('Best Cost')
plt.title('PSO Parameter Sensitivity: Best Cost vs. Inertia Weight (w)')
plt.legend()
plt.show()



# Find the set of parameters that resulted in the minimum cost
best_result = min(results, key=lambda x: x[0])

print(f"Best cost: {best_result[0]}")
print(f"Optimal c1: {best_result[1]}")
print(f"Optimal c2: {best_result[2]}")
print(f"Optimal w: {best_result[3]}")
#save all the results in a csv file to a path /Users/sheidasaffari/Documents/Research/Papers/Paper2/Codes/results/Sensitivity analysis
df_results.to_csv('/Users/sheidasaffari/Documents/Research/Papers/Paper2/Codes/results/Sensitivity analysis/PSO_SensitivityAnalysis.csv', index=False)


######################END################################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################
# --------------------------
# Save Optimization Results
# --------------------------
# def save_optimization_results(file_path, voxel_size, num_cameras, W1, W2, W3, iteration,
#                               num_particles, options, optimal_camera_poses, total_voxels,
#                               camera_coverage_counts, coverage_percentages, num_voxels_covered,
#                               avg_distance_diff, avg_angle, distance_diffs, angles):
#     current_datetime_for_filename = datetime.now().strftime('%d%b_%H-%M')
#     file_path = f"{file_path.rstrip('/')}/optimization_results_{current_datetime_for_filename}.csv"
#     current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     rows = []
#     rows.append(['Date and Time', current_datetime])
#     rows.extend([
#         ["Voxel Size", voxel_size],
#         ["Number of Cameras", num_cameras],
#         ["W1", W1],
#         ["W2", W2],
#         ["W3", W3],
#         ["Iteration", iteration],
#         ["Number of Particles", num_particles],
#         ["c1", options['c1']],
#         ["c2", options['c2']],
#         ["w", options['w']]
#     ])
#     for i, (pose, coverage, percentage) in enumerate(zip(optimal_camera_poses, camera_coverage_counts, coverage_percentages), start=1):
#         rows.append([f'Camera {i} Best Pose', str(pose)])
#         rows.append([f'Camera {i} Coverage Count', coverage])
#         rows.append([f'Camera {i} Coverage Percentage', percentage])
#     rows.append(['Total Voxels Covered', num_voxels_covered])
#     rows.append(['Total Voxels', total_voxels])
#     rows.append(['Average Pairwise Distance Difference', avg_distance_diff])
#     rows.append(['Average Pairwise Angle (deg)', avg_angle])
#     rows.append(['Full Pairwise Distance Differences', np.array2string(distance_diffs, precision=2)])
#     rows.append(['Full Pairwise Angles (deg)', np.array2string(angles, precision=2)])
#     df = pd.DataFrame(rows, columns=['Parameter', 'Value'])
#     df.to_csv(file_path, index=False)
#     print(f"Optimization results saved to {file_path}")

# base_path = '/Users/sheidasaffari/Documents/Research/Papers/Paper2/Codes/results'

# save_optimization_results(base_path, voxel_size, num_cameras, W1, W2, W3, iteration, num_particles,
#                             options, optimal_camera_poses, total_voxels, camera_coverage_counts,
#                             coverage_percentages, num_voxels_covered, avg_distance_diff, avg_angle,
#                             distance_diffs, angles)

# # Optional: visualize the optimized cameras and their FOVs
# visualize_cameras_and_fov(optimal_camera_poses)
