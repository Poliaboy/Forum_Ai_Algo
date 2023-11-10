import random
import matplotlib.pyplot as plt


# Fonction pour l'évaluation de la qualité d'une solution (fitness)
def evaluate_fitness(x):
    return x ** 2


# Fonction pour la recherche locale des abeilles employées
def local_search(bee, neighbourhood_size=1):
    # Explorer les positions voisines
    candidates = [bee + i for i in range(-neighbourhood_size, neighbourhood_size + 1) if i != 0]
    for new_pos in candidates:
        if evaluate_fitness(new_pos) < evaluate_fitness(bee):
            return new_pos  # Nouvelle position avec une meilleure fitness
    return bee  # Pas d'amélioration, rester à la position actuelle


# Fonction pour la recherche globale des abeilles butineuses
def global_search(search_range):
    # Sélectionner une position aléatoire dans la plage de recherche spécifiée
    return random.uniform(-search_range, search_range)


# Fonction pour la mise à jour des informations de la meilleure solution, COMMUNICATION
def update_best_solution(bees, best_solution):
    for bee in bees:
        if evaluate_fitness(bee) < evaluate_fitness(best_solution):
            best_solution = bee
    return best_solution


# Algorithme d'Optimisation par Colonie d'Abeilles (ABC)
def abc_optimization(num_bees=10, max_iters=100, search_range=10):
    # Initialisation des abeilles à des positions aléatoires
    bees = [random.uniform(-search_range, search_range) for _ in range(num_bees)]
    best_solution = bees[0]  # Initialiser la meilleure solution
    best_solution = update_best_solution(bees, best_solution)

    # Itérations de l'algorithme
    for _ in range(max_iters):
        # Phase de recherche des abeilles employées
        for i in range(len(bees)):
            bees[i] = local_search(bees[i])

        # Mise à jour de la meilleure solution
        best_solution = update_best_solution(bees, best_solution)

        # Phase de recherche des butineuses
        for i in range(len(bees)):
            # Si l'abeille n'a pas trouvé de meilleure solution, elle devient une butineuse
            if bees[i] == best_solution:  # Supposons que seules les meilleures abeilles restent employées
                continue
            bees[i] = global_search(search_range)

        # Mise à jour de la meilleure solution
        best_solution = update_best_solution(bees, best_solution)

    return best_solution, evaluate_fitness(best_solution)


def abc_optimization_tracking(num_bees=10, max_iters=10, search_range=10):
    # Initialisation des abeilles à des positions aléatoires
    bees = [random.uniform(-search_range, search_range) for _ in range(num_bees)]
    best_solution = bees[0]  # Initialiser la meilleure solution
    best_solution = update_best_solution(bees, best_solution)

    # Pour enregistrer l'évolution des positions des abeilles à chaque itération
    steps_tracking = []

    # Itérations de l'algorithme
    for iteration in range(max_iters):
        # Enregistrer la position des abeilles
        steps_tracking.append(bees.copy())

        # Phase de recherche des abeilles employées
        for i in range(len(bees)):
            bees[i] = local_search(bees[i])

        # Mise à jour de la meilleure solution
        best_solution = update_best_solution(bees, best_solution)

        # Phase de recherche des butineuses
        for i in range(len(bees)):
            # Si l'abeille n'a pas trouvé de meilleure solution, elle devient une butineuse
            if bees[i] == best_solution:  # Supposons que seules les meilleures abeilles restent employées
                continue
            bees[i] = global_search(search_range)

        # Mise à jour de la meilleure solution
        best_solution = update_best_solution(bees, best_solution)

    return best_solution, evaluate_fitness(best_solution), steps_tracking


# Exécuter l'algorithme ABC avec suivi
best_x, best_fx, tracking = abc_optimization_tracking(num_bees=5, max_iters=10, search_range=10)

# Maintenant, nous allons créer des visualisations pour chaque itération
for i, step in enumerate(tracking):
    plt.figure(figsize=(10, 5))
    plt.scatter(step, [evaluate_fitness(x) for x in step], color='blue', label='Bees')
    plt.plot([-5, 5], [evaluate_fitness(best_x), evaluate_fitness(best_x)], color='red', label='Best Solution')
    plt.title(f'Iteration {i + 1}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim(-5, 5)
    plt.ylim(0, 25)
    plt.legend()
    plt.grid(True)
    plt.show()
