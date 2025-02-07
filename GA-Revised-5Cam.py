import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Point, box
import random
import open3d as o3d
from itertools import cycle
import csv

##Parameters set by teh user
# Room and target box dimensions
# Target object can randomly locate at any point in the target box
room_size = {'width': 5.181, 'length': 5.308, 'height': 2.700}
box_size = {'width': 3.581, 'length': 3.708, 'height': .700}
box_position = {'x': (room_size['width'] - box_size['width']) / 2,
                'y': (room_size['length'] - box_size['length']) / 2,
                'z': 0}

box_center = {
    'x': box_position['x'] + box_size['width'] / 2,
    'y': box_position['y'] + box_size['length'] / 2,
    'z': box_position['z'] + box_size['height'] / 2
} 
camera_fov = {'horizontal': 40, 'vertical': 60}
h_fov = 40
v_fov = 60

max_depth = 4
voxel_size = 0.2

# In this stage we define the number of cameras, later this can be one objective parameter to be found by the optimization algorithm
num_cameras= 2

# Function to check if a point is within the room but outside the box
def is_valid_position(x, y, z, room_dim = room_size, box_cord = box_position, box_dim = box_size):
    if box_cord['z'] <= z <= (box_cord['z'] + box_dim['height']):
                # Define 2D polygons for the room and the box based on their x and y dimensions
        room_poly = box(0, 0, room_dim['width'], room_dim['length'])
        box_poly = box(box_cord['x'], box_cord['y'], 
                    box_cord['x'] + box_dim['width'], 
                    box_cord['y'] + box_dim['length'])
        
        # Calculate the difference between the room and the box to define the valid search space
        search_space = room_poly.difference(box_poly)
        
        # Create a 2D point from the x and y coordinates
        point = Point(x, y)
        
        # Check if the point is within the 2D search space and the z coordinate is within the valid height range
        is_in_2d_space = search_space.contains(point)
        return is_in_2d_space
    elif (box_cord['z'] + box_dim['height']) < z <= room_dim['height']:
        return True
    else: 
        return False

    # is_in_height_range = box_cord['z'] <= z <= (box_cord['z'] + box_dim['height'])
    # The position is valid if it's within the 2D search space and the specified height range
    # return is_in_2d_space and is_in_height_range


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



def voxel_positions(box_size, voxel_size):
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

    # Camera distance to the plane containing the point (new_depth for obtaining 4 corners)
    dist_to_plane = np.sqrt(distance_2_camera**2 - distance_2_optical_line**2)

    # 🛠️ **Fix: Add `roll` in function call**
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


global voxel_coord
voxel_coord = voxel_positions(box_size, voxel_size)


def camera_voxel_matrix(camera_poses, pitches, yaws, rolls, points):
    camera_voxel_matrix = np.zeros((len(camera_poses), len(points)))
    # print('camera_voxel_matrix:',camera_voxel_matrix)
    for i, camera_pos in enumerate(camera_poses):
        for j, point in enumerate(points):
            # camera_voxel_matrix[i, j] = Is_inside_pyramid(point, camera_pos, camera_fov['horizontal'], camera_fov['vertical'], max_depth, pitches[i], yaws[i], rolls[i])
            camera_voxel_matrix[i, j] = Is_inside_pyramid(
    point, camera_pos, 
    camera_fov['horizontal'], camera_fov['vertical'], max_depth, 
    pitches[i], yaws[i], rolls[i] if i < len(rolls) else 0
)

    return camera_voxel_matrix


def fitness_function(chromosome):

    camera_poses = [pose[:3] for pose in chromosome]  # (x, y, z)
    pitches = [pose[3] for pose in chromosome]  # Pitch
    yaws = [pose[4] for pose in chromosome]  # Yaw
    rolls = [pose[5] if len(pose) > 5 else 0 for pose in chromosome]  # Extract roll if present, otherwise 0

    big_matrix = camera_voxel_matrix(camera_poses, pitches, yaws, rolls, voxel_coord)


    # calculate sum of columns of the camera voxel matrix 
    sum_columns = np.sum(big_matrix, axis=0)
    for i , voxel_sum in enumerate(sum_columns):
        if voxel_sum:
            sum_columns[i] = 1
    # Calculate the number of voxels covered by at least one camera
    num_voxels_covered = np.sum(sum_columns)
    # print('num_voxels_covered:',num_voxels_covered)
    normalized_num_voxels_covered = num_voxels_covered / len(voxel_coord)
    normalized_fitness_score = normalized_num_voxels_covered
    
    return normalized_fitness_score


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
    flat_params = np.zeros(num_cameras * 6)

    for i in range(num_cameras):

        flat_params[i*6:i*6+3] = camera_poses[i]
        flat_params[i*6+3] = pitches[i]
        flat_params[i*6+4] = yaws[i]
        flat_params[i*6+5] = 0  # Default roll to 0


    return flat_params


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

    # Ensure angles are within [-180, 180] range
    yaw_deg = (yaw_deg + 180) % 360 - 180
    pitch_deg = (pitch_deg + 180) % 360 - 180

    return pitch_deg, yaw_deg


def create_initial_positions(num_cameras, num_particles, min_bounds, max_bounds):
    # Number of parameters per camera (x, y, z, pitch, yaw)
    num_params_per_camera = 6
    
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
            pitch, yaw = calculate_orientation(camera_pos, target_pos)
            
            # Assign positions and angles for the current camera
            # init_pos[i, idx_offset:idx_offset + num_params_per_camera] = [x, y, z, pitch, yaw]
            init_pos[i, idx_offset:idx_offset + 6] = [x, y, z, pitch, yaw, 0]  # Include roll

    
    return init_pos


def standardize_camera_solution(solution):
    standardized_solution = []
    for camera in solution:
        if isinstance(camera, np.ndarray):
            # Convert numpy array to tuple
            camera_tuple = tuple(camera)
        elif isinstance(camera, tuple):
            camera_tuple = camera
        else:
            raise TypeError("Camera configuration must be a tuple or numpy array.")
        standardized_solution.append(camera_tuple)
    return standardized_solution

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
        pitch, yaw = camera[3:5]
        fov_corners = calculate_fov_pyramid(camera_pos, h_fov, v_fov, max_depth, pitch, yaw)
        
        # FOV lines visualization
        for corner in fov_corners:
            ax.plot([camera_pos[0], corner[0]], [camera_pos[1], corner[1]], [camera_pos[2], corner[2]], color=color)
        
        # Base rectangle visualization
        base_x = [corner[0] for corner in fov_corners] + [fov_corners[0][0]]
        base_y = [corner[1] for corner in fov_corners] + [fov_corners[0][1]]
        base_z = [corner[2] for corner in fov_corners] + [fov_corners[0][2]]
        ax.plot(base_x, base_y, base_z, color=color)
        
        # Camera position
        ax.scatter(*camera_pos, color=color, s=20, label=f'Camera {color}')
    
    ax.scatter(*[box_center['x'], box_center['y'], 0], color='black', s=20, label='Target Box Center')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Placement and FOV Visualization')
    ax.legend()

    plt.show()

def initialize_population_with_orientation(pop_size, num_cameras, min_bounds, max_bounds):
    # Use the create_initial_positions function to generate initial positions and orientations
    init_pos = create_initial_positions(num_cameras, pop_size, min_bounds, max_bounds)
    # Convert the initial positions array into the desired population format
    population = []
    for i in range(pop_size):
        chromosome = []
        for j in range(num_cameras):
            idx_offset = j * 6  # 5 parameters per camera: x, y, z, pitch, yaw
            x, y, z, pitch, yaw = init_pos[i, idx_offset:idx_offset + 5]
            roll = 0 
            chromosome.append((x, y, z, pitch, yaw, roll))
        population.append(chromosome)
    return population

# Define the bounds for the optimization algorithm
param_bounds = [
    (0, room_size['width']),  # x bound
    (0, room_size['length']),  # y bound
    (0, room_size['height']),  # z bound
    (-40, 40),  # pitch bound
    (-40,40),  # yaw bound
    (-1, 1)  # roll bound
]

# Flatten the bounds for all cameras
min_bounds = np.array([bound[0] for bound in param_bounds] * num_cameras)
max_bounds = np.array([bound[1] for bound in param_bounds] * num_cameras)


#selecting parents for GA
def select_parents(population, fitness_fn, num_parents):
    # Select the best individuals based on the fitness function
    fitness_values = [fitness_fn(chromosome) for chromosome in population]
    # Sort the population based on the fitness values
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
    # Select the best parents
    parents = sorted_population[:num_parents]
    return parents



def crossover(parents, offspring_size):
    offspring = []
    for i in range(offspring_size):
        # Select two parents
        parent1 = parents[i % len(parents)]
        parent2 = parents[(i + 1) % len(parents)]
        
        # Randomly select a crossover point
        crossover_point = np.random.randint(1, len(parent1))

        # Convert slices to lists before concatenation
        offspring_part1 = list(parent1[:crossover_point])  # Convert to list
        offspring_part2 = list(parent2[crossover_point:])  # Convert to list
        
        # Combine both parts as a list
        offspring.append(offspring_part1 + offspring_part2)

    return offspring

def crossover_with_probability(parents, offspring_size, crossover_probability):
    offspring = []
    for i in range(offspring_size):
        # With probability crossover_probability perform crossover,
        # otherwise randomly choose one parent.
        if random.random() < crossover_probability:
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)]
            crossover_point = np.random.randint(1, len(parent1))
            offspring_part1 = list(parent1[:crossover_point])
            offspring_part2 = list(parent2[crossover_point:])
            offspring.append(offspring_part1 + offspring_part2)
        else:
            # Simply copy one parent (or select randomly)
            offspring.append(random.choice(parents))
    return offspring


# def mutate(offspring_crossover, mutation_rate):
#     for i in range(len(offspring_crossover)):
#         # Convert the tuple to a list before mutating
#         offspring_list = list(offspring_crossover[i])  

#         # Ensure the mutation index is within bounds
#         mutation_point = np.random.randint(0, len(offspring_list))

#         # Mutation value
#         mutation_value = np.random.uniform(-1.0, 1.0, 1).item()  # Ensure a scalar float

#         # Convert the target tuple to a list, mutate, then convert it back
#         mutated_camera = list(offspring_list[mutation_point])
#         mutated_camera[np.random.randint(0, len(mutated_camera))] += mutation_value  
#         offspring_list[mutation_point] = tuple(mutated_camera)

#         # Convert the list back to a tuple before reassigning
#         offspring_crossover[i] = tuple(offspring_list)

#     return offspring_crossover

def mutate(offspring_crossover, mutation_rate):
    # offspring_crossover is a list of chromosomes (each is a tuple of camera configurations)
    mutated_offspring = []
    for chromosome in offspring_crossover:
        # Convert the chromosome (tuple of cameras) to a list so we can modify it
        chromosome_list = list(chromosome)
        # Loop over each camera configuration in the chromosome
        new_chromosome = []
        for camera in chromosome_list:
            # Convert the camera tuple to a list to mutate individual parameters
            camera_list = list(camera)
            # Loop over each gene in the camera configuration
            for gene_idx in range(len(camera_list)):
                # With probability equal to mutation_rate, mutate this gene
                if np.random.rand() < mutation_rate:
                    mutation_value = np.random.uniform(-1.0, 1.0)
                    camera_list[gene_idx] += mutation_value
            # Convert back to tuple and add to new chromosome
            new_chromosome.append(tuple(camera_list))
        mutated_offspring.append(tuple(new_chromosome))
    return mutated_offspring

#Genetic algorithm
def genetic_algorithm(population, fitness_fn, num_generations):
    # Number of parents to select
    num_parents = 5
    # Number of offspring to produce
    num_offspring = len(population) - num_parents
    # Crossover probability
    crossover_probability = 0.8
    # Mutation probability
    mutation_rate = 0.1
    # Record the best fitness value at each generation
    best_fitness_values = []
    # Record the best solution at each generation
    best_solutions = []
    # Run the genetic algorithm for a set number of generations
    for generation in range(num_generations):
        # Select the best parents
        parents = select_parents(population, fitness_fn, num_parents)
        # Create the next generation through crossover
        # offspring_crossover = crossover(parents, num_offspring)
        offspring_crossover = crossover_with_probability(parents, num_offspring, crossover_probability)
        # Apply mutation
        offspring_mutation = mutate(offspring_crossover, mutation_rate)
        # Add the parents and offspring to the new population
        population = parents + offspring_mutation
        # Record the best fitness value at each generation
        fitness_values = [fitness_fn(chromosome) for chromosome in population]
        best_fitness = max(fitness_values)
        best_solution = population[fitness_values.index(best_fitness)]
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Save best_fitness_values to CSV
    with open('fitness_over_generations.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Best Fitness'])
        for i, fitness in enumerate(best_fitness_values):
            writer.writerow([i, fitness])
            
    return best_solution, best_fitness




# Initialize the population with positions and orientations
np.random.seed(11)
pop_size = 100
population = initialize_population_with_orientation(pop_size, num_cameras, min_bounds, max_bounds)
# print('population:',population)

#Starting the genetic algorithm
# Define the number of generations
num_generations = 50
# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm(population, fitness_function, num_generations)
print('best_solution:',best_solution)
print('best_fitness:',best_fitness)

# Save the generation number and fitness values as csv file


standardized_best_solution = standardize_camera_solution(best_solution)
# Now, visualize the standardized best solution
# visualize_cameras_and_fov_pso(room_size, box_position, box_size, standardized_best_solution)
best_solution, best_fitness, final_population = genetic_algorithm(pop_size, num_cameras, num_generations)
