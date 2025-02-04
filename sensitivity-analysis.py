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