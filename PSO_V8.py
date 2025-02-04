## V5 GA Code

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Point, box
import random
import open3d as o3d
from pyswarm import pso
from scipy.special import comb
from pyswarms.utils.plotters import plot_cost_history
import pyswarms as ps
from itertools import cycle
from datetime import datetime
import torch
from torch.autograd import Variable


# Parameters set by the user
# Room and target box dimensions
# Target object can randomly locate at any point in the target box
room_size = {'width': 5.181, 'length': 5.308, 'height': 2.700}
box_size = {'width': 3.581, 'length': 3.708, 'height': .700}
box_position = {'x': (room_size['width'] - box_size['width']) / 2,
                'y': (room_size['length'] - box_size['length']) / 2,
                'z': 0}

# Room dimensions
room_width = room_size['width']
room_length = room_size['length']
room_height = room_size['height']

# Center of the floor
center_of_floor = np.array([room_width / 2, room_length / 2, 0])


box_center = {
    'x': box_position['x'] + box_size['width'] / 2,
    'y': box_position['y'] + box_size['length'] / 2,
    'z': box_position['z'] + box_size['height'] / 2
} 

##Can later have different FOV angles for different cameras if needed
camera_fov = {'horizontal': 40, 'vertical': 60}
max_depth = 4  # Maximum depth of FOV  
voxel_size = 0.2
num_params_per_camera = 6
num_cameras = 4


# Cost Function Coefficients
W1 = 0.4   # Coefficient of Coevarge
W2 = 0.2 # Coefficient of Resolution (Scale)
W3 = 0.4  # Coefficient of Viewpoints Variety


# Function to check if a point is within the room but outside the box

def is_valid_position(x, y, z, room_dim = room_size, box_cord = box_position, box_dim = box_size):   
    # Create a 2D point from the x and y coordinates
    point = Point(x, y)

    # Define 2D polygons for the room and the box based on their x and y dimensions
    room_poly = box(0, 0, room_dim['width'], room_dim['length'])
    box_poly = box(box_cord['x'], box_cord['y'], 
                box_cord['x'] + box_dim['width'], 
                box_cord['y'] + box_dim['length'])

    if box_cord['z'] <= z <= (box_cord['z'] + box_dim['height']):

        
        # Calculate the difference between the room and the box to define the valid search space
        search_space = room_poly.difference(box_poly)
        
        # Check if the point is within the 2D search space and the z coordinate is within the valid height range
        is_in_2d_space = search_space.contains(point)
        return is_in_2d_space
    
    elif ((box_cord['z'] + box_dim['height']) < z <= room_dim['height']) and (room_poly.contains(point)):
        return True
    
    else:
        return False


def calculate_fov_pyramid_old(camera_pos, h_fov, v_fov, max_depth, pitch, yaw):
    half_h_fov = math.radians(h_fov / 2)
    half_v_fov = math.radians(v_fov / 2)

    # Calculate distances from the camera to the rectangle edges
    dx = max_depth * math.tan(half_h_fov)
    dy = max_depth * math.tan(half_v_fov)

    # Calculate base corners
    corners = [
        [camera_pos[0] - dx, camera_pos[1] - dy, camera_pos[2] - max_depth],
        [camera_pos[0] + dx, camera_pos[1] - dy, camera_pos[2] - max_depth],
        [camera_pos[0] + dx, camera_pos[1] + dy, camera_pos[2] - max_depth],
        [camera_pos[0] - dx, camera_pos[1] + dy, camera_pos[2] - max_depth]
    ]

    # Convert pitch and yaw from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    # Correctly align rotation matrices for a Y-up coordinate system
    R_yaw = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                      [0, 1, 0],
                      [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])

    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                        [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

    # Apply rotations to corners, ensuring the correct order
    rotated_corners = []
    for corner in corners:
        corner_array = np.array(corner)
        # Apply pitch then yaw rotation to the corner points
        rotated_corner = np.dot(R_yaw, np.dot(R_pitch, corner_array - np.array(camera_pos))) + np.array(camera_pos)
        rotated_corners.append(rotated_corner.tolist())
    
    return rotated_corners

def calculate_fov_pyramid(camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll):
    half_h_fov = math.radians(h_fov / 2)
    half_v_fov = math.radians(v_fov / 2)

    # Calculate distances from the camera to the rectangle edges
    dx = max_depth * math.tan(half_h_fov)
    dy = max_depth * math.tan(half_v_fov)

    # Calculate base corners
    corners = [
        [camera_pos[0] - dx, camera_pos[1] - dy, camera_pos[2] - max_depth],
        [camera_pos[0] + dx, camera_pos[1] - dy, camera_pos[2] - max_depth],
        [camera_pos[0] + dx, camera_pos[1] + dy, camera_pos[2] - max_depth],
        [camera_pos[0] - dx, camera_pos[1] + dy, camera_pos[2] - max_depth]
    ]

    # Convert pitch, yaw, and roll from degrees to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([[np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                      [0, 1, 0],
                      [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]])

    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                        [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

    R_roll = np.array([[np.cos(roll_rad), -np.sin(roll_rad), 0],
                       [np.sin(roll_rad), np.cos(roll_rad), 0],
                       [0, 0, 1]])

    # Apply rotations to corners, ensuring the correct order of roll, pitch, then yaw
    rotated_corners = []
    for corner in corners:
        corner_array = np.array(corner)
        # Subtract camera_pos to apply rotation around the camera, then add it back
        rotated_corner = np.dot(R_yaw, np.dot(R_pitch, np.dot(R_roll, corner_array - np.array(camera_pos)))) + np.array(camera_pos)
        rotated_corners.append(rotated_corner.tolist())
    
    return rotated_corners

def voxel_positions(boxox_size, vel_size):
    # Generate voxel positions within the box
    voxels = []
    for x in np.arange(box_position['x'], box_position['x'] + box_size['width'], voxel_size):
        for y in np.arange(box_position['y'], box_position['y'] + box_size['length'], voxel_size):
            for z in np.arange(box_position['z'], box_position['z'] + box_size['height'], voxel_size):
                center_x = x + voxel_size / 2.0
                center_y = y + voxel_size / 2.0
                center_z = z + voxel_size / 2.0
                voxels.append([center_x, center_y, center_z])
    return voxels

global voxel_coord
voxel_coord = voxel_positions(box_size, voxel_size)

def Is_inside_pyramid(voxel, camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll):
    # Calculate the floor corners of the FOV pyramid
    floor_corners = calculate_fov_pyramid(camera_pos, h_fov, v_fov, max_depth, pitch, yaw, roll)

    # Initialize sum_corners as a numpy array of zeros
    sum_corners = np.array([0.0, 0.0, 0.0])
    
    # Sum up the coordinates of the floor corners
    for corner in floor_corners:
        sum_corners += np.array(corner)

    # Calculate the center of the floor corners
    center_floor = sum_corners / len(floor_corners)

    # Camera position should be a numpy array for consistent operations
    camera_pos_array = np.array(camera_pos)

    # Calculating the optical line vector using two points (camera_pos and center_floor)
    optical_line_vector = center_floor - camera_pos_array

    # Point of interest
    point = np.array(voxel)

    # Calculating the vector from camera position to the point
    point_vector = point - camera_pos_array

    # Ensure optical_line_vector is normalized for accurate projection calculation
    optical_line_vector_normalized = optical_line_vector / np.linalg.norm(optical_line_vector)

    # Calculate the projection length accurately
    projection_length = np.dot(point_vector, optical_line_vector_normalized)

    if projection_length < 0 or projection_length > max_depth:
        return False

    # Use the normalized optical line vector for calculating the closest point
    closest_point_on_line = camera_pos_array + projection_length * optical_line_vector_normalized

    # Calculate the distance between the given point and the closest point on the line (which is perpendicular)
    distance_2_optical_line = np.linalg.norm(point - closest_point_on_line)
  
    # Distance between the camera and the point
    distance_2_camera = np.linalg.norm(point_vector)

    # camera distance to the plane containing the point (new_depth for obtaining 4 corners)
    dist_to_plane = np.sqrt(distance_2_camera**2 - distance_2_optical_line**2)

    point_plane_corners = calculate_fov_pyramid(camera_pos, h_fov, v_fov, dist_to_plane, pitch, yaw, roll)

    numpy_corners = [np.array(corner) for corner in point_plane_corners]

    # Calculate the normal of the plane using the cross product of two edges of the rectangle
    edge1 = numpy_corners[1] - numpy_corners[0]
    edge2 = numpy_corners[2] - numpy_corners[1]
    plane_normal = np.cross(edge1, edge2)

    def is_on_right_side(test_point, corner1, corner2, normal):
        edge = corner2 - corner1
        point_vector = test_point - corner1
        cross_product = np.cross(edge, point_vector)
        # Checking if the cross product is in the same direction as the plane normal
        return np.dot(cross_product, normal) >= 0

    inside = True
    for i in range(len(numpy_corners)):
        next_index = (i + 1) % len(numpy_corners)  # Ensure loop back to the first corner
        if not is_on_right_side(point, numpy_corners[i], numpy_corners[next_index], plane_normal):
            inside = False
            break

    return inside

def camera_voxel_matrix(camera_poses, pitches, yaws, rolls, points):
    camera_voxel_matrix = np.zeros((len(camera_poses), len(points)))
    # print('camera_voxel_matrix:',camera_voxel_matrix)
    for i, camera_pos in enumerate(camera_poses):
        for j, point in enumerate(points):
            camera_voxel_matrix[i, j] = Is_inside_pyramid(point, camera_pos, camera_fov['horizontal'], camera_fov['vertical'], max_depth, pitches[i], yaws[i], rolls[i])
    return camera_voxel_matrix

def camera_distance_differences(camera_poses, points):
    # Calculate the number of combinations of two cameras from the total number of cameras
    num_camera_combinations = len(camera_poses) * (len(camera_poses) - 1) // 2
    # Initialize the array to hold distance differences for each camera pair and each point
    distance_diffs = np.zeros((num_camera_combinations, len(points)))
    
    combination_index = 0  # To track the index for each camera pair combination
    for i in range(len(camera_poses)):
        for j in range(i + 1, len(camera_poses)):
            for k, point in enumerate(points):
                # Calculate the distance from each camera to the point
                distance_from_camera_i = np.linalg.norm(np.array(point) - np.array(camera_poses[i]))
                distance_from_camera_j = np.linalg.norm(np.array(point) - np.array(camera_poses[j]))
                # Calculate the absolute difference in distances
                distance_diffs[combination_index, k] = abs(distance_from_camera_i - distance_from_camera_j)
            combination_index += 1
    
    return distance_diffs

def camera_angles(camera_poses, points):
    angles = []
    # Iterate through combinations of two camera poses
    for i in range(len(camera_poses)):
        for j in range(i + 1, len(camera_poses)):
            camera_vector_angles = []
            for point in points:
                # Calculate vectors from cameras to point
                vector_a = np.array(point) - np.array(camera_poses[i])
                vector_b = np.array(point) - np.array(camera_poses[j])
                
                # Calculate the angle between vectors
                cos_angle = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
                angle = np.arccos(cos_angle) * (180 / np.pi)  # Convert radians to degrees
                
                camera_vector_angles.append(angle)
                
            angles.append(camera_vector_angles)
            
    return np.array(angles)

def flatten_camera_params(camera_poses, pitches, yaws):
    """
    Flatten camera parameters into a single array.
    
    Parameters:
    - camera_poses: Array of camera positions, shape (num_cameras, 3)
    - pitches: Array of camera pitches, shape (num_cameras,)
    - yaws: Array of camera yaws, shape (num_cameras,)
    
    Returns:
    - flat_params: Flattened array of all camera parameters
    """
    num_cameras = len(camera_poses)
    flat_params = np.zeros(num_cameras * num_params_per_camera) #Revised based on new code : Considering Roll
    # flat_params = np.zeros(num_cameras * 5)
    for i in range(num_cameras):
        flat_params[i*5:i*5+3] = camera_poses[i]
        flat_params[i*5+3] = pitches[i]
        flat_params[i*5+4] = yaws[i]

    return flat_params

def visualize_cameras_and_fov_pso(room_size, box_position, box_size, cameras_solution): 
    fig = plt.figure(figsize=(12, 10)) 
    ax = fig.add_subplot(111, projection='3d') 

 # Visualize room 
    room_corners = [[0, 0, 0],  
                    [room_size['width'], 0, 0], 
                    [room_size['width'], room_size['length'], 0],  
                    [0, room_size['length'], 0], 
                    [0, 0, room_size['height']],  
                    [room_size['width'], 0, room_size['height']], 
                    [room_size['width'], room_size['length'], room_size['height']],  
                    [0, room_size['length'], room_size['height']]] 
    room_edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle 
                  (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle 
                  (0, 4), (1, 5), (2, 6), (3, 7)]  # Side edges 
    for edge in room_edges: 
        start_point, end_point = room_corners[edge[0]], room_corners[edge[1]] 
        ax.plot3D(*zip(start_point, end_point), color="cyan") 
 
    # Visualize box 
    box_min = [box_position['x'], box_position['y'], box_position['z']] 
    box_max = [box_position['x'] + box_size['width'], box_position['y'] + box_size['length'], box_position['z'] + box_size['height']] 
    box_corners = [[box_min[0], box_min[1], box_min[2]],  
                   [box_max[0], box_min[1], box_min[2]],  
                   [box_max[0], box_max[1], box_min[2]],  
                   [box_min[0], box_max[1], box_min[2]],  
                   [box_min[0], box_min[1], box_max[2]],  
                   [box_max[0], box_min[1], box_max[2]],  
                   [box_max[0], box_max[1], box_max[2]],  
                   [box_min[0], box_max[1], box_max[2]]] 
    box_edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle 
                 (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle 
                 (0, 4), (1, 5), (2, 6), (3, 7)]  # Side edges 
    for edge in box_edges: 
        start_point, end_point = box_corners[edge[0]], box_corners[edge[1]] 
        ax.plot3D(*zip(start_point, end_point), color="green") 
    # Prepare a cycle of colors for different cameras
    colors = cycle(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta'])
    
    # Visualize cameras and FOVs
    for camera, color in zip(cameras_solution, colors):
        camera_pos = camera[:3]
        pitch, yaw, roll = camera[3:num_params_per_camera]
        fov_corners = calculate_fov_pyramid(camera_pos, camera_fov['horizontal'], camera_fov['vertical'], max_depth, pitch, yaw, roll)
        
        # FOV lines visualization
        for corner in fov_corners:
            ax.plot([camera_pos[0], corner[0]], [camera_pos[1], corner[1]], [camera_pos[2], corner[2]], color=color)
        
        # Base rectangle visualization
        base_x = [corner[0] for corner in fov_corners] + [fov_corners[0][0]]
        base_y = [corner[1] for corner in fov_corners] + [fov_corners[0][1]]
        base_z = [corner[2] for corner in fov_corners] + [fov_corners[0][2]]
        ax.plot(base_x, base_y, base_z, color=color)
        
        # Camera position
        ax.scatter(*camera_pos, color=color, s=10, label=f'Camera {color}')
    
    ax.scatter(*[box_center['x'], box_center['y'], 0], color='black', s=10, label='Target Box Center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Placement and FOV Visualization')
    ax.legend()

    plt.show()


def pso_fitness_function(particles):
    """
    PSO-compatible fitness function that accepts a 2D array of parameters.
    Each row in the array represents the flattened parameters for all cameras of a single particle.
    
    Parameters:
    - particles: A 2D array where each row contains all camera parameters (position, pitch, yaw) sequentially for one particle.
    
    Returns:
    - A 1D array of fitness scores, where each score corresponds to a particle in the swarm.
    """
    # Initialize an array to store the fitness score for each particle
    fitness_scores = np.zeros(particles.shape[0])
    penalty = 0

    for i, flat_params in enumerate(particles):
        num_cameras = len(flat_params) // num_params_per_camera  # Each camera has 5 parameters
        camera_poses = flat_params.reshape((num_cameras, num_params_per_camera))[:, :3]  # Extract 3D positions
        pitches = flat_params.reshape((num_cameras, num_params_per_camera))[:, 3]  # Extract pitch angles
        yaws = flat_params.reshape((num_cameras, num_params_per_camera))[:, 4]  # Extract yaw angles
        rolls = flat_params.reshape((num_cameras, num_params_per_camera))[:, 5]  # Extract roll angles

        # Proceed with the calculations as before
        # Note: Ensure voxel_coord and other variables are accessible within this function
        big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)
        sum_columns = np.clip(np.sum(big_matrix, axis=0), 0, 1)
        num_voxels_covered = np.sum(sum_columns)

        distance_differences = camera_distance_differences(camera_poses, voxel_coord)
        sum_distance_differences = np.sum(distance_differences)

        angles = camera_angles(camera_poses, voxel_coord)
        sum_angles = np.sum(angles)

        # Calculate normalized scores based on the specific objectives
        normalized_num_voxels_covered = num_voxels_covered / len(voxel_coord)
        normalized_distance_differences = sum_distance_differences / (comb(num_cameras, 2) * len(voxel_coord) * np.sqrt(room_size['width']**2 + room_size['length']**2 + room_size['height']**2))
        normalized_angles = sum_angles / (comb(num_cameras, 2) * len(voxel_coord) * 180)

        for pos in camera_poses:
            if is_inside_box(pos, box_position, box_size):
                penalty += 1000  # Penalize positions inside the box

        # Combine objectives into a single fitness score for this particle
        normalized_fitness_score =  W1 * normalized_num_voxels_covered + W2 * normalized_distance_differences +  W3 * normalized_angles
        normalized_fitness_score = normalized_num_voxels_covered

        # Store the negative of the score since PSO minimizes the function
        fitness_scores[i] = -normalized_fitness_score

    return fitness_scores + penalty

def is_inside_box(camera_position, box_position, box_size):
    """Check if the camera position is inside the box."""
    x, y, z = camera_position
    x_min, x_max = box_position['x'], box_position['x'] + box_size['width']
    y_min, y_max = box_position['y'], box_position['y'] + box_size['length']
    z_min, z_max = box_position['z'], box_position['z'] + box_size['height']
    
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max


def calculate_orientation(camera_pos, target_pos):
    """
    Calculate orientation (pitch and yaw) for a camera to look towards a target position,
    compatible with the way rotations are applied in calculate_fov_pyramid function,
    but with the directions inverted to ensure the camera faces towards the target.
    """
    # Vector from camera to target
    vector_to_target = np.array(target_pos) - np.array(camera_pos)
    dx, dy, dz = vector_to_target

    # Yaw calculation: angle between camera-target vector projection on the XZ plane and the Z-axis
    yaw = math.atan2(dx, dz)
    yaw_deg = math.degrees(yaw) + 180  # Invert direction

    # Pitch calculation: angle between camera-target vector and its projection on the XZ plane
    dist_xz = math.sqrt(dx**2 + dz**2)
    pitch = math.atan2(-dy, dist_xz)  # Negative because pitch up should be positive
    pitch_deg = math.degrees(pitch) * -1  # Invert direction

    roll_deg = 0  # Assume no roll for simplicity

    # Ensure angles are within [-180, 180] range
    yaw_deg = (yaw_deg + 180) % 360 - 180
    pitch_deg = (pitch_deg + 180) % 360 - 180
    roll_deg = (roll_deg + 180) % 360 -180 

    return pitch_deg, yaw_deg, roll_deg

def create_initial_positions(num_cameras, num_particles, min_bounds, max_bounds):
 
    # Initialize the positions array
    init_pos = np.zeros((num_particles, num_cameras * num_params_per_camera))
    
    for i in range(num_particles):
        for j in range(num_cameras):
            # Calculate index offset for current camera
            idx_offset = j * num_params_per_camera
            
            # Random position within bounds
            x = np.random.uniform(min_bounds[idx_offset], max_bounds[idx_offset])
            y = np.random.uniform(min_bounds[idx_offset + 1], max_bounds[idx_offset + 1])
            z = np.random.uniform(min_bounds[idx_offset + 2], max_bounds[idx_offset + 2])

            # Ensure the camera is outside the box
            while is_inside_box([x, y, z], box_position, box_size):
                x = np.random.uniform(min_bounds[idx_offset], max_bounds[idx_offset])
                y = np.random.uniform(min_bounds[idx_offset + 1], max_bounds[idx_offset + 1])
                z = np.random.uniform(min_bounds[idx_offset + 2], max_bounds[idx_offset + 2])

            camera_pos = np.array([x, y, z])
            target_pos = np.array([box_center['x'], box_center['y'], 0])
            # Assigining the orientations toward the target object 
            pitch, yaw, roll = calculate_orientation(camera_pos, target_pos)
            init_pos[i, idx_offset:idx_offset + num_params_per_camera] = [x, y, z, pitch, yaw, roll]
    
    return init_pos

param_bounds = [
    (0, room_size['width']),  # x bound
    (0, room_size['length']),  # y bound
    (0, room_size['height']),  # z bound
    (-180, 180),  # pitch bound
    (-180,180),  # yaw bound
    (-180,180)  # roll bound
]

# ####################################################
# ############## PSO with pyswarms ###################
# ####################################################

## Set the device to GPU if available, otherwise print GPU not available and use CPU
# Define the function to check GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define constants
maxiter = 500
num_particles = 50
iteration = 10
np.random.seed(220)

# Define camera-related parameters (adjust as needed)


# Compute dimensions based on the number of cameras
dimensions = 6 * num_cameras

# Expand bounds for all cameras
min_bounds = [b[0] for b in param_bounds] * num_cameras
max_bounds = [b[1] for b in param_bounds] * num_cameras

# Convert the bounds to NumPy arrays
min_bounds_array = np.array(min_bounds)
max_bounds_array = np.array(max_bounds)

# Create initial positions
init_pos = np.random.uniform(low=min_bounds_array, high=max_bounds_array, size=(num_particles, dimensions))

# Convert to PyTorch tensor and move to GPU
init_pos_tensor = torch.tensor(init_pos, dtype=torch.float32, device=device)

# PSO options
options = {'c1': 1.2, 'c2': 0.7, 'w': 0.3075}

# Initialize the optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                    dimensions=dimensions,
                                    options=options,
                                    bounds=(min_bounds_array, max_bounds_array),
                                    init_pos=init_pos_tensor.cpu().numpy())  # Convert tensor to NumPy array

# Perform optimization
cost, pos = optimizer.optimize(pso_fitness_function, iters=iteration)

# Reshape the optimal positions back to the structured format
optimal_camera_poses = pos.reshape((-1, 6))
print("Optimized Camera Poses:", optimal_camera_poses)

# Assuming voxel_coord and camera_voxel_matrix functions are defined elsewhere
total_voxels = len(voxel_coord)

# Extract camera poses, pitches, yaws, and rolls
camera_poses = optimal_camera_poses[:, :3]
pitches = optimal_camera_poses[:, 3]
yaws = optimal_camera_poses[:, 4]
rolls = optimal_camera_poses[:, 5]

# Calculate the big matrix indicating voxel coverage by each camera
big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)

# Calculate the number of voxels each camera covers
camera_coverage_counts = np.sum(big_matrix, axis=1)
print('Camera Coverage Counts:', camera_coverage_counts)

# Calculate coverage percentage for each camera
coverage_percentages = (camera_coverage_counts / total_voxels) * 100
print("Coverage Percentages for each camera:", coverage_percentages)

# Calculate the total number of voxels covered by all cameras
sum_columns = np.clip(np.sum(big_matrix, axis=0), 0, 1)
num_voxels_covered = np.sum(sum_columns)
print('Total Number of Voxels Covered:', num_voxels_covered)


#########################################################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################
# To run this part of the code, above lines where we are doing
# the optimization and calculating teh cost and best pose should be commented
# For the purpose of computational resources, we used only 5 values in teh space, 
# then we will limit this bound and find more optimum hyperparameetrs

# iteration = 10
# # Set the seed for reproducibility
# np.random.seed(220)

# # Define the parameter grid
# w_values = np.linspace(0.01, 0.8, num = 5)
# c1_values = np.linspace(1, 3, num= 5)
# c2_values = np.linspace(1, 3, num= 5)

# results  = []



# for c1 in c1_values:
#     for c2 in c2_values:
#         for w in w_values:
#             # Update the options with the current set of parameters
#             options = {'c1': c1, 'c2': c2, 'w': w}

#             # Initialize the optimizer with the current set of parameters
#             optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
#                                                 dimensions=dimensions,
#                                                 options=options,
#                                                 bounds=(min_bounds, max_bounds),
#                                                 init_pos=init_pos)
            
#             # Perform optimization
#             cost, pos = optimizer.optimize(pso_fitness_function, iters=iteration)
            
#             # Store the results along with the corresponding parameters
#             results.append((cost, c1, c2, w))


# # Convert results to a DataFrame for easier analysis
# df_results = pd.DataFrame(results, columns=['best_cost', 'c1', 'c2', 'w'])


# # Example: Print the top 5 parameter combinations with the lowest cost
# print(df_results.sort_values(by='best_cost').head())

# # You could also create plots to visualize the sensitivity
# # For instance, plotting the best cost for different values of w
# plt.figure(figsize=(10, 6))
# for c1 in c1_values:
#     for c2 in c2_values:
#         subset = df_results[(df_results['c1'] == c1) & (df_results['c2'] == c2)]
#         plt.plot(subset['w'], subset['best_cost'], marker='o', label=f'c1={c1}, c2={c2}')

# plt.xlabel('Inertia Weight (w)')
# plt.ylabel('Best Cost')
# plt.title('PSO Parameter Sensitivity: Best Cost vs. Inertia Weight (w)')
# plt.legend()
# plt.show()



# # Find the set of parameters that resulted in the minimum cost
# best_result = min(results, key=lambda x: x[0])

# print(f"Best cost: {best_result[0]}")
# print(f"Optimal c1: {best_result[1]}")
# print(f"Optimal c2: {best_result[2]}")
# print(f"Optimal w: {best_result[3]}")


######################END################################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################

##Save optimization results


def save_optimization_results(file_path, voxel_size, num_cameras, W1, W2, W3, iteration, num_particles, options, optimal_camera_poses, total_voxels, camera_coverage_counts, coverage_percentages, num_voxels_covered):
    # Format current datetime for the filename
    current_datetime_for_filename = datetime.now().strftime('%d%b_%H-%M')
    # Adjust file_path to include dynamically generated datetime
    file_path = f"{file_path.rstrip('/')}/optimization_results_{current_datetime_for_filename}.csv"
    
    # Current datetime for inclusion in the CSV
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Initialize an empty list to store data
    rows = []
    # Append general information
    rows.append(['Date and Time', current_datetime])
    rows.extend([
        ["Voxel Size", voxel_size],
        ["Number of Cameras", num_cameras],
        ["W1", W1],
        ["W2", W2],
        ["W3", W3],
        ["Iteration", iteration],
        ["Number of Particles", num_particles],
        ["c1", options['c1']],
        ["c2", options['c2']],
        ["w", options['w']]
    ])

    # Append camera specific data
    for i, (pose, coverage, percentage) in enumerate(zip(optimal_camera_poses, camera_coverage_counts, coverage_percentages), start=1):
        rows.append([f'Camera {i} Best Pose', str(pose)])
        rows.append([f'Camera {i} Coverage Count', coverage])
        rows.append([f'Camera {i} Coverage Percentage', percentage])

    # Append total voxels covered and total voxels
    rows.append(['Total Voxels Covered', num_voxels_covered])
    rows.append(['Total Voxels', total_voxels])

    # Convert rows to DataFrame
    df = pd.DataFrame(rows, columns=['Parameter', 'Value'])

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Optimization results saved to {file_path}")

base_path = '/home/sheida/Downloads/Research/Camera Placement/Results'  # Base directory for saving the file
save_optimization_results(base_path, voxel_size, num_cameras, W1, W2, W3, iteration, num_particles, options, optimal_camera_poses, total_voxels, camera_coverage_counts, coverage_percentages, num_voxels_covered)

