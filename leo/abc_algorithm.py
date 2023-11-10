import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


# Define the sphere function as the objective function
def objective_function(solution):
    return sum(x ** 2 for x in solution)


# Initialize the population (food sources)
def initialize_population(pop_count, solution_size, lower_bound, upper_bound):
    population = []
    for _ in range(pop_count):
        solution = [random.uniform(lower_bound, upper_bound) for _ in range(solution_size)]
        population.append((solution, objective_function(solution)))
    return population


# Employed bee phase
def employed_bee_phase(population, lower_bound, upper_bound):
    new_population = []
    for solution, fitness in population:
        # Generate a new solution near the current one
        k = random.randint(0, len(solution) - 1)
        phi = random.uniform(-1, 1)
        new_solution = solution[:]
        new_solution[k] = solution[k] + phi * (solution[k] - random.choice(population)[0][k])
        # Apply bounds
        new_solution[k] = min(max(new_solution[k], lower_bound), upper_bound)
        new_fitness = objective_function(new_solution)
        # Greedy selection
        if new_fitness < fitness:
            new_population.append((new_solution, new_fitness))
        else:
            new_population.append((solution, fitness))
    return new_population


# Onlooker bee phase
def onlooker_bee_phase(population):
    # Calculate the fitness of each solution
    fitnesses = [1 / (1 + fitness) for _, fitness in population]
    max_fitness = max(fitnesses)
    probabilities = [fitness / max_fitness for fitness in fitnesses]

    new_population = []
    for i in range(len(population)):
        if random.random() < probabilities[i]:
            new_population.append(population[i])
        else:
            new_population.append(random.choice(population))

    return new_population


# Scout bee phase
def scout_bee_phase(population, lower_bound, upper_bound, solution_size, limit):
    new_population = []
    for solution, fitness in population:
        if fitness > limit:
            # Replace with a new random solution
            new_solution = [random.uniform(lower_bound, upper_bound) for _ in range(solution_size)]
            new_fitness = objective_function(new_solution)
            new_population.append((new_solution, new_fitness))
        else:
            new_population.append((solution, fitness))
    return new_population


# ABC algorithm
def abc_algorithm(pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit):
    # Initialization
    population = initialize_population(pop_count, solution_size, lower_bound, upper_bound)
    best_solution = min(population, key=lambda x: x[1])[0]
    best_fitness = min(population, key=lambda x: x[1])[1]

    # Main loop
    for _ in range(max_iterations):
        population = employed_bee_phase(population, lower_bound, upper_bound)
        population = onlooker_bee_phase(population)
        population = scout_bee_phase(population, lower_bound, upper_bound, solution_size, limit)

        # Update the best solution found so far
        current_best_solution = min(population, key=lambda x: x[1])[0]
        current_best_fitness = min(population, key=lambda x: x[1])[1]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution

    return best_solution, best_fitness


# Parameters
pop_count = 30
solution_size = 2  # 2-dimensional problem
lower_bound = -5
upper_bound = 5
max_iterations = 50
limit = 5  # Abandonment limit (number of trials after which a source is abandoned)

# Run the ABC algorithm
best_solution, best_fitness = abc_algorithm(pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit)
best_solution, best_fitness


# Define the Rosenbrock function
def rosenbrock_function(solution):
    x, y = solution
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


# Modify the ABC algorithm to use the Rosenbrock function
def abc_algorithm_rosenbrock(pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit):
    # Initialization
    population = initialize_population(pop_count, solution_size, lower_bound, upper_bound)
    best_solution = min(population, key=lambda x: x[1])[0]
    best_fitness = min(population, key=lambda x: x[1])[1]

    # To visualize the progression
    solutions_progress = [best_solution]

    # Main loop
    for _ in range(max_iterations):
        population = employed_bee_phase(population, lower_bound, upper_bound)
        population = onlooker_bee_phase(population)
        population = scout_bee_phase(population, lower_bound, upper_bound, solution_size, limit)

        # Update the best solution found so far
        current_best_solution = min(population, key=lambda x: x[1])[0]
        current_best_fitness = min(population, key=lambda x: x[1])[1]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            solutions_progress.append(best_solution)

    return best_solution, best_fitness, solutions_progress


# Run the ABC algorithm on the Rosenbrock function
solution_size = 2  # Rosenbrock is a 2D function
best_solution, best_fitness, solutions_progress = abc_algorithm_rosenbrock(
    pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit
)

# Prepare to visualize the Rosenbrock function
x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
x, y = np.meshgrid(x, y)
z = rosenbrock_function((x, y))

# Plot the Rosenbrock function in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=0.8)

# Plot the solutions found by the ABC algorithm
solutions_x = [solution[0] for solution in solutions_progress]
solutions_y = [solution[1] for solution in solutions_progress]
solutions_z = [rosenbrock_function(solution) for solution in solutions_progress]
ax.scatter(solutions_x, solutions_y, solutions_z, color='r', s=50)

# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('ABC Optimization on Rosenbrock Function')

# Show the plot
plt.show()

best_solution, best_fitness


# Define a simplified objective function for the cantilever beam optimization
def beam_objective_function(solution):
    x, y = solution  # x and y are the dimensions of the beam's cross-section
    mass_coefficient = 1.0  # This would be related to the material density and beam length
    deflection_coefficient = 1.0  # This accounts for material properties and force applied
    mass = x * y  # Assuming uniform material, mass is proportional to the cross-sectional area
    deflection = 1 / (x * y ** 3)  # Simplified deflection formula for a cantilever beam
    return mass_coefficient * mass + deflection_coefficient * deflection


# Modify the ABC algorithm to use the new objective function
def abc_algorithm_beam(pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit):
    # Override the initialization to use the beam objective function
    def initialize_population(pop_count, solution_size, lower_bound, upper_bound):
        population = []
        for _ in range(pop_count):
            solution = [random.uniform(lower_bound, upper_bound) for _ in range(solution_size)]
            population.append((solution, beam_objective_function(solution)))
        return population

    # Initialization
    population = initialize_population(pop_count, solution_size, lower_bound, upper_bound)
    best_solution = min(population, key=lambda x: x[1])[0]
    best_fitness = min(population, key=lambda x: x[1])[1]

    # To visualize the progression
    solutions_progress = [best_solution]

    # Main loop
    for _ in range(max_iterations):
        population = employed_bee_phase(population, lower_bound, upper_bound)
        population = onlooker_bee_phase(population)
        population = scout_bee_phase(population, lower_bound, upper_bound, solution_size, limit)

        # Update the best solution found so far
        current_best_solution = min(population, key=lambda x: x[1])[0]
        current_best_fitness = min(population, key=lambda x: x[1])[1]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            solutions_progress.append(best_solution)

    return best_solution, best_fitness, solutions_progress


# Run the ABC algorithm on the beam optimization problem
solution_size = 2  # We have two dimensions to optimize: width (x) and height (y) of the beam
lower_bound = 0.1  # Avoid zero to prevent infinite deflection
upper_bound = 5.0  # Some upper limit on the dimensions of the beam
max_iterations = 50
limit = 5  # Abandonment limit (number of trials after which a source is abandoned)

best_solution, best_fitness, solutions_progress = abc_algorithm_beam(
    pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit
)

# Prepare to visualize the objective function
x = np.linspace(lower_bound, upper_bound, 100)
y = np.linspace(lower_bound, upper_bound, 100)
x, y = np.meshgrid(x, y)
z = beam_objective_function((x, y))

# Plot the objective function in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=0.8)

# Plot the solutions found by the ABC algorithm
solutions_x = [solution[0] for solution in solutions_progress]
solutions_y = [solution[1] for solution in solutions_progress]
solutions_z = [beam_objective_function(solution) for solution in solutions_progress]
ax.scatter(solutions_x, solutions_y, solutions_z, color='r', s=50)

# Set labels and title
ax.set_xlabel('Width (x)')
ax.set_ylabel('Height (y)')
ax.set_zlabel('Objective Value')
ax.set_title('ABC Optimization of a Cantilever Beam')


def abc_algorithm_beam_progress(pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit,
                                progress_interval):
    def initialize_population(pop_count, solution_size, lower_bound, upper_bound):
        population = []
        for _ in range(pop_count):
            solution = [random.uniform(lower_bound, upper_bound) for _ in range(solution_size)]
            population.append((solution, beam_objective_function(solution)))
        return population

    # Initialization
    population = initialize_population(pop_count, solution_size, lower_bound, upper_bound)
    best_solution = min(population, key=lambda x: x[1])[0]
    best_fitness = min(population, key=lambda x: x[1])[1]

    # To visualize the progression
    solutions_progress = {0: best_solution}  # Store initial best solution

    # Main loop
    for iteration in range(max_iterations):
        population = employed_bee_phase(population, lower_bound, upper_bound)
        population = onlooker_bee_phase(population)
        population = scout_bee_phase(population, lower_bound, upper_bound, solution_size, limit)

        # Update the best solution found so far
        current_best_solution = min(population, key=lambda x: x[1])[0]
        current_best_fitness = min(population, key=lambda x: x[1])[1]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            if iteration % progress_interval == 0 or iteration == max_iterations - 1:
                solutions_progress[iteration] = best_solution  # Store progress at intervals

    return best_solution, best_fitness, solutions_progress


# New parameter for visualization intervals
progress_interval = 10  # We will visualize the solution every 10 iterations

# Run the ABC algorithm with progress tracking
best_solution, best_fitness, solutions_progress = abc_algorithm_beam_progress(
    pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit, progress_interval
)

# Visualization of the objective function at intervals
fig = plt.figure(figsize=(18, 10))
plot_number = 1
total_plots = len(solutions_progress)

for iteration, solution in solutions_progress.items():
    ax = fig.add_subplot(2, (total_plots + 1) // 2, plot_number, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=0.8, antialiased=False)

    # Plot the current best solution
    ax.scatter(solution[0], solution[1], beam_objective_function(solution), color='r', s=50)

    # Set labels and title
    ax.set_xlabel('Width (x)')
    ax.set_ylabel('Height (y)')
    ax.set_zlabel('Objective Value')
    ax.set_title(f'Iteration {iteration}')

    plot_number += 1

# Adjust the layout
plt.tight_layout()
plt.show()

print(best_solution, best_fitness)
