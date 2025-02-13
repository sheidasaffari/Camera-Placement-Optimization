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
from scipy.stats.qmc import LatinHypercube
import os


######################Start##############################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################
#LatinHypercube = LatinHypercube
#this part of the code is for sensitivity analysis of PSO hyperparameters using Latin Hypercube Sampling
#The hyperparameters are c1, c2, and w
#The optimization function is PSO-camera-opt.py
#The results are saved in a csv file


iteration = 100
# Set the seed for reproducibility
np.random.seed(18)

num_samples = 500 # Number of samples for LHS

# Define LHS sampler for 3 variables (c1, c2, w)
#d = 3 means we have 3 hyperparameters

sampler = LatinHypercube(d=3)
lhs_samples = sampler.random(n=num_samples)

# Scale samples to the desired parameter ranges
c1_values = 0.5 + lhs_samples[:, 0] * (2.0 - 0.5)  # Scale to [0.5, 2.0]
c2_values = 0.5 + lhs_samples[:, 1] * (2.0 - 0.5)  # Scale to [0.5, 2.0]
w_values = 0.2 + lhs_samples[:, 2] * (1.0 - 0.2)   # Scale to [0.2, 1.0]


results  = []



for i in range(num_samples):
    c1 = c1_values[i]
    c2 = c2_values[i]
    w = w_values[i]

    # Define PSO options for the current sample
    options = {'c1': c1, 'c2': c2, 'w': w}

    # Run PSO optimization
    optimizer = ps.single.GlobalBestPSO(n_particles=num_particles,
                                        dimensions=dimensions,
                                        options=options,
                                        bounds=(min_bounds, max_bounds),
                                        init_pos=init_pos)

    cost, pos = optimizer.optimize(pso_fitness_function, iters=iteration)

    # Store results
    results.append({
        'best_cost': cost,
        'c1': c1,
        'c2': c2,
        'w': w,
        'num_cameras': num_cameras,
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'iterations': iteration
    })

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save results

output_dir = "/Users/sheidasaffari/Documents/Research/Papers/Paper2/Codes/results/sen_analysis"
output_path = os.path.join(output_dir, "PSO_SensitivityAnalysis.csv")

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame
df_results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Print the best results
best_result = df_results.sort_values(by='best_cost').iloc[0]
print(f"Best cost: {best_result['best_cost']}")
print(f"Optimal c1: {best_result['c1']}")
print(f"Optimal c2: {best_result['c2']}")
print(f"Optimal w: {best_result['w']}")

######################END################################
#####Sensitivity Analysis of PSO Hyperparameters#########
#########################################################
