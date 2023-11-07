import random

# parameters
num_bees = 10
num_iterations = 50
max_trials = 5 # maximum attempt for each bee


# function to optimize : in this case f(x) =  x^2
def objective_function(x):
    return x**2

# initialize population with random positions
positions = [random.uniform(-10,10) for i in range(num_bees)]


for iteration in range(num_iterations):
    
    for i in range(num_bees):
        current_bee_position = positions[i]
        current_bee_value = objective_function(current_bee_position)
        
        # search new positions for the bees
        for j in range(max_trials):
            new_bee_position = current_bee_position + random.uniform(-1,1)
            new_bee_value = objective_function(new_bee_position)
            
            if new_bee_value < current_bee_value:
                 current_bee_position = new_bee_position
                 current_bee_value = new_bee_value
        
        # Mise à jour de la position de l'abeille employée
        positions[i] = current_bee_position
    
    
    # find the best solution to this iteration
    best_solution = min(positions, key=objective_function)
    best_value = objective_function(best_solution)
    print(f"Itération {iteration + 1}: Meilleure solution = {best_solution}, Valeur = {best_value}")