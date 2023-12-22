import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock_function(x, y):
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

# Parameters for ABC Algorithm
num_bees = 50
num_iterations = 100
search_range = 0.5  # Range for neighborhood search in x and y
x_range = (-2, 2)
y_range = (-1, 3)

# Initialize bees at random positions in 3D space (x, y)
np.random.seed(0)
bee_positions_x = np.random.uniform(x_range[0], x_range[1], num_bees)
bee_positions_y = np.random.uniform(y_range[0], y_range[1], num_bees)

# Visualization setup for Rosenbrock function
x_grid, y_grid = np.meshgrid(np.linspace(x_range[0], x_range[1], 400),
                             np.linspace(y_range[0], y_range[1], 400))
z_values = rosenbrock_function(x_grid, y_grid)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_values, cmap='viridis', alpha=0.8)
ax.set_title('ABC Algorithm on Rosenbrock Function')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# ABC Algorithm Simulation for Rosenbrock function
for iteration in range(num_iterations):
    for i in range(num_bees):
        # Explore new position near current one
        new_x = bee_positions_x[i] + np.random.uniform(-search_range, search_range)
        new_y = bee_positions_y[i] + np.random.uniform(-search_range, search_range)
        # Move to better position
        if rosenbrock_function(new_x, new_y) < rosenbrock_function(bee_positions_x[i], bee_positions_y[i]):
            bee_positions_x[i] = new_x
            bee_positions_y[i] = new_y

    # Plot bees' positions at certain iterations for visualization
    if iteration in [10, 20, 30, 40, 49]:
        ax.scatter(bee_positions_x, bee_positions_y, rosenbrock_function(bee_positions_x, bee_positions_y),
                   label=f'Iteration {iteration+1}', s=20)

# Final best position
best_idx = np.argmin(rosenbrock_function(bee_positions_x, bee_positions_y))
best_position_x = bee_positions_x[best_idx]
best_position_y = bee_positions_y[best_idx]
ax.scatter(best_position_x, best_position_y, rosenbrock_function(best_position_x, best_position_y),
           color='red', s=100, label='Best Position')

# Add legend
ax.legend()
plt.show()
