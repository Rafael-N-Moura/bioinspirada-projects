import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class EightQueensGA:
    def __init__(self):
        self.population_size = 10
        self.crossover_prob = 0.9
        self.mutation_prob = 0.4
        self.max_evaluations = 10000
        self.board_size = 8
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluations = 0

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            individual = list(range(self.board_size))
            random.shuffle(individual)
            self.population.append(individual)

    def calculate_fitness(self, individual: List[int]) -> int:
        conflicts = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                if abs(i - j) == abs(individual[i] - individual[j]):
                    conflicts += 1
        return conflicts

    def select_parents(self) -> Tuple[List[int], List[int]]:
        candidates = random.sample(self.population, 5)
        candidates.sort(key=self.calculate_fitness)
        return candidates[0], candidates[1]

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        size = len(parent1)
        point = random.randint(0, size-1)

        def create_child(p1: List[int], p2: List[int], cut_point: int) -> List[int]:
            child = [-1] * size
            child[:cut_point] = p1[:cut_point]
            remaining = [x for x in p2 if x not in child[:cut_point]]
            child[cut_point:] = remaining
            return child

        child1 = create_child(parent1, parent2, point)
        child2 = create_child(parent2, parent1, point)
        return child1, child2

    def mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_prob:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def select_survivors(self, offspring: List[List[int]], strategy: str = "+") -> None:
        if strategy == "+":
            combined = self.population + offspring
            combined.sort(key=self.calculate_fitness)
            self.population = combined[:self.population_size]
        elif strategy == ",":
            offspring.sort(key=self.calculate_fitness)
            self.population = offspring[:self.population_size]

    def run(self, survivor_strategy: str = "+") -> Tuple[bool, int, List[float], List[float]]:
        self.initialize_population()
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluations = 0
        iteration = 0

        while self.evaluations < self.max_evaluations:
            population_fitness = [self.calculate_fitness(ind) for ind in self.population]
            self.evaluations += len(self.population)

            self.fitness_history.append(np.mean(population_fitness))
            self.best_fitness_history.append(min(population_fitness))

            if min(population_fitness) == 0:
                return True, iteration, self.fitness_history, self.best_fitness_history

            offspring = []
            for _ in range(self.population_size * 7 // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])

            self.select_survivors(offspring, strategy=survivor_strategy)
            iteration += 1

        return False, iteration, self.fitness_history, self.best_fitness_history

def run_experiment(num_runs: int = 30, survivor_strategy: str = "+") -> None:
    converged_runs = 0
    iterations_to_converge = []
    final_fitness_values = []

    for run in range(num_runs):
        print(f"Executando experimento {run + 1}/{num_runs}")
        ga = EightQueensGA()
        converged, iterations, _, _ = ga.run(survivor_strategy)

        if converged:
            converged_runs += 1
            iterations_to_converge.append(iterations)

        final_fitness = min([ga.calculate_fitness(ind) for ind in ga.population])
        final_fitness_values.append(final_fitness)

    print("\nResultados do experimento:")
    print(f"Número de execuções que convergiram: {converged_runs}/{num_runs}")
    if iterations_to_converge:
        print(f"Média de iterações para convergência: {np.mean(iterations_to_converge):.2f}")
        print(f"Desvio padrão das iterações: {np.std(iterations_to_converge):.2f}")
    print(f"Fitness médio final: {np.mean(final_fitness_values):.2f}")
    print(f"Desvio padrão do fitness final: {np.std(final_fitness_values):.2f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ga.fitness_history, label='Fitness Médio')
    plt.plot(ga.best_fitness_history, label='Melhor Fitness')
    plt.xlabel('Iteração')
    plt.ylabel('Fitness')
    plt.title('Convergência do Algoritmo')
    plt.legend()
    plt.grid(True)
    plt.show()


# run_experiment(survivor_strategy="+")  # para (μ + λ)
# run_experiment(survivor_strategy=",")  # para (μ, λ)


if __name__ == "__main__":
    run_experiment(survivor_strategy=",")
