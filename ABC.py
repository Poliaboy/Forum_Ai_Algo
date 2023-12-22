import random
import numpy as np


class ABC_Algorithm:
    def __init__(self, objective_function, pop_count, solution_size, lower_bound, upper_bound, max_iterations, limit):
        self.objective_function = objective_function
        self.pop_count = pop_count
        self.solution_size = solution_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iterations = max_iterations
        self.limit = limit
        self.population = None
        self.best_solution = None
        self.best_fitness = None
        self.progress = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_count):
            solution = [random.uniform(self.lower_bound[i], self.upper_bound[i]) for i in range(self.solution_size)]
            fitness = self.objective_function(solution)
            self.population.append((solution, fitness))

    def employed_bee_phase(self):
        new_population = []
        for index, (solution, fitness) in enumerate(self.population):
            # Select a solution different from the current one
            partner_index = index
            while partner_index == index:
                partner_index = np.random.randint(self.pop_count)
            partner_solution = self.population[partner_index][0]

            # Modify solution by moving towards or away from partner solution
            new_solution = np.copy(solution)
            for k in range(self.solution_size):
                phi = np.random.uniform(-1, 1)
                new_solution[k] += phi * (solution[k] - partner_solution[k])
                new_solution[k] = np.clip(new_solution[k], self.lower_bound[k], self.upper_bound[k])

            new_fitness = self.objective_function(new_solution)
            if new_fitness < fitness:
                new_population.append((new_solution, new_fitness))
            else:
                new_population.append((solution, fitness))
        self.population = new_population

    def onlooker_bee_phase(self):
        fitnesses = [1 / (1 + fitness) if fitness >= 0 else 1 + abs(fitness) for _, fitness in self.population]
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]

        new_population = []
        for _ in range(self.pop_count):
            r = random.random()
            chosen_index = 0
            for i, probability in enumerate(probabilities):
                r -= probability
                if r <= 0:
                    chosen_index = i
                    break

            solution, fitness = self.population[chosen_index]
            k = random.randint(0, self.solution_size - 1)
            phi = random.uniform(-1, 1)
            new_solution = solution[:]
            new_solution[k] = solution[k] + phi * (solution[k] - random.choice(self.population)[0][k])
            new_solution[k] = min(max(new_solution[k], self.lower_bound[k]), self.upper_bound[k])
            new_fitness = self.objective_function(new_solution)
            if new_fitness < fitness:
                new_population.append((new_solution, new_fitness))
            else:
                new_population.append((solution, fitness))

        self.population = new_population

    def scout_bee_phase(self):
        new_population = []
        for solution, fitness in self.population:
            if fitness > self.limit:
                new_solution = [random.uniform(self.lower_bound[i], self.upper_bound[i]) for i in
                                range(self.solution_size)]
                new_fitness = self.objective_function(new_solution)
                new_population.append((new_solution, new_fitness))
            else:
                new_population.append((solution, fitness))
        self.population = new_population

    def run(self):
        self.initialize_population()
        self.best_solution = min(self.population, key=lambda x: x[1])[0]
        self.best_fitness = min(self.population, key=lambda x: x[1])[1]

        for iteration in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            # Save the best solution found so far
            current_best_solution, current_best_fitness = min(self.population, key=lambda x: x[1])
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution

            # Track progress
            self.progress.append((self.best_solution.copy(), self.best_fitness.copy()))

        return self.best_solution, self.best_fitness, self.progress
