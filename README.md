# Camera Configuration Optimization for In-Lab Benchmark Data Collection

## Overview

This project implements an optimization framework for camera placement in a laboratory setting. Our goal is to maximize coverage of a target space (voxelized 3D environment), diversify viewpoints, and achieve varied scale images. The optimization problem is approached as a multiobjective task, and we use both Particle Swarm Optimization (PSO) and Genetic Algorithms (GA) to explore a continuous search space for the best camera configurations.

## Project Description

The core idea behind the project is to position cameras in a room (while avoiding a designated target box) so that the data collection process achieves:
- **High Coverage:** Maximizing the number of voxels (small cubic cells) in the target space that are captured.
- **Viewpoint Diversity:** Maximizing the angular separation between cameras.
- **Scale Variability:** Maximizing the differences in camera-to-voxel distances across camera pairs.

The project includes:
- **Geometry and Search Space Modeling:** Voxelization of the target space, modeling of each camera's field of view (FOV) as a 3D pyramid using rotation matrices, and functions for determining valid camera positions.
- **PSO and GA Implementations:** Custom fitness functions that combine normalized metrics (coverage, pairwise distance differences, and pairwise angles) and enforce physical constraints (e.g., cameras not being inside the target box).
- **Visualization:** Functions to visualize the room, target box, camera positions, and their FOVs in 3D.
- **Results Recording:** Generation of CSV reports containing key performance metrics (e.g., coverage percentage, average pairwise distance, and average pairwise angle).

## Installation

### Prerequisites
- Python 3.7 or later
- The following Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `shapely`
  - `open3d`
  - `torch`
  - `pyswarms`
  - `scipy`

### Installing Dependencies
You can install the required dependencies using `pip`. For example, create a `requirements.txt` file with:

numpy pandas matplotlib shapely open3d torch pyswarms scipy



Then run: 
pip install -r requirements.txt



## Usage

### Running the PSO Optimization

1. **Configure Parameters:**  
   Open the main Python script and adjust parameters in the **User-Defined Parameters** section (e.g., room dimensions, number of cameras, and fitness function weights).

2. **Execute the Script:**  
   Run the script from the command line:


python pso_optimization.py


The script will:
- Initialize valid camera positions.
- Run the PSO optimization.
- Plot the cost history (fitness vs. iteration).
- Save the optimization results in a CSV file.
- Visualize the optimized camera positions and their FOVs in a 3D plot.

### Running the GA Optimization

A separate module or branch may contain the GA implementation. Follow similar instructions:
python ga_optimization.py


## Understanding the Results

- **Fitness Metrics:**  
The PSO/GA fitness function combines normalized coverage, distance differences, and angular diversity. The ideal combined normalized score depends on your weighting:
- For instance, if using W1 = W2 = W3 = 1, the perfect solution might achieve a fitness score near -3 (since the optimizer minimizes the negative value).

- **CSV Reports:**  
The results file (e.g., `optimization_results_04Feb_14-30.csv`) contains details such as:
- Voxel size, number of cameras, and fitness weights.
- Best camera poses and their coverage counts.
- Average pairwise distance differences and angles.

## Convergence & Parameter Tuning

If the optimization appears to be stuck (e.g., cost plateauing), consider:
- Increasing the number of iterations.
- Adjusting the PSO hyperparameters (`c1`, `c2`, and `w`).
- Refining the normalization factors in the fitness function.

For multiobjective optimization, trade-offs between coverage and viewpoint/scale diversity might require careful tuning.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

