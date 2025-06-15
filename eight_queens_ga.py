import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
from enum import Enum

class MutationType(Enum):
    SWAP = "swap"  # Troca de dois genes
    INSERTION = "insertion"  # Inserção de um gene próximo a outro
    INVERSION = "inversion"  # Inversão de uma sub-string
    PERTURBATION = "perturbation"  # Rearranjo aleatório de um subconjunto

class EightQueensGA:
    def __init__(self, mutation_type: MutationType = MutationType.SWAP):
        self.population_size = 10
        self.crossover_prob = 0.9
        self.mutation_prob = 0.4
        self.max_evaluations = 10000
        self.board_size = 8
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluations = 0
        self.mutation_type = mutation_type

    def initialize_population(self):
        """Inicializa a população com permutações aleatórias."""
        self.population = []
        for _ in range(self.population_size):
            individual = list(range(self.board_size))
            random.shuffle(individual)
            self.population.append(individual)

    def calculate_fitness(self, individual: List[int]) -> int:
        """Calcula o fitness de um indivíduo (número de conflitos)."""
        conflicts = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                # Verifica conflitos na mesma diagonal
                if abs(i - j) == abs(individual[i] - individual[j]):
                    conflicts += 1
        return conflicts

    def select_parents(self) -> Tuple[List[int], List[int]]:
        """Seleciona dois pais usando o método de ranking."""
        candidates = random.sample(self.population, 5)
        candidates.sort(key=self.calculate_fitness)
        return candidates[0], candidates[1]

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Realiza o crossover cut-and-crossfill."""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        size = len(parent1)
        point = random.randint(0, size-1)
        
        def create_child(p1: List[int], p2: List[int], cut_point: int) -> List[int]:
            child = [-1] * size
            # Copia a primeira parte do primeiro pai
            child[:cut_point] = p1[:cut_point]
            # Preenche o resto com elementos do segundo pai na ordem correta
            remaining = [x for x in p2 if x not in child[:cut_point]]
            child[cut_point:] = remaining
            return child

        child1 = create_child(parent1, parent2, point)
        child2 = create_child(parent2, parent1, point)
        return child1, child2

    def mutate_swap(self, individual: List[int]) -> List[int]:
        """
        Mutação por troca: dois genes são escolhidos aleatoriamente e suas posições são trocadas.
        Este operador preserva a maior parte da adjacência da informação.
        Esperado: Boa preservação de padrões locais, mas pode ser limitado em explorar novas configurações.
        """
        if random.random() < self.mutation_prob:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def mutate_insertion(self, individual: List[int]) -> List[int]:
        """
        Mutação por inserção: dois genes são escolhidos aleatoriamente, o segundo é movido para próximo do primeiro.
        Este operador preserva a maior parte da ordem da informação adjacente.
        Esperado: Melhor preservação de sequências ordenadas, útil quando a ordem relativa dos genes é importante.
        """
        if random.random() < self.mutation_prob:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            gene = individual.pop(idx2)
            # Insere o gene próximo ao primeiro índice escolhido
            insert_pos = (idx1 + 1) % len(individual)
            individual.insert(insert_pos, gene)
        return individual

    def mutate_inversion(self, individual: List[int]) -> List[int]:
        """
        Mutação por inversão: dois genes são escolhidos aleatoriamente e a sub-string entre eles é invertida.
        Este operador preserva a adjacência da informação, mas modifica a ordem.
        Esperado: Bom para explorar novas configurações mantendo conexões locais, útil para quebrar padrões rígidos.
        """
        if random.random() < self.mutation_prob:
            idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
            individual[idx1:idx2+1] = reversed(individual[idx1:idx2+1])
        return individual

    def mutate_perturbation(self, individual: List[int]) -> List[int]:
        """
        Mutação por perturbação: um subconjunto de genes é escolhido aleatoriamente e rearranjado.
        Este operador permite mudanças mais drásticas na configuração.
        Esperado: Maior exploração do espaço de busca, mas pode perder boas características locais.
        """
        if random.random() < self.mutation_prob:
            # Escolhe um subconjunto aleatório de genes
            subset_size = random.randint(2, len(individual) // 2)
            subset_indices = random.sample(range(len(individual)), subset_size)
            subset = [individual[i] for i in subset_indices]
            random.shuffle(subset)
            # Recoloca os genes rearranjados
            for i, idx in enumerate(subset_indices):
                individual[idx] = subset[i]
        return individual

    def mutate(self, individual: List[int]) -> List[int]:
        """Aplica a mutação selecionada ao indivíduo."""
        if self.mutation_type == MutationType.SWAP:
            return self.mutate_swap(individual)
        elif self.mutation_type == MutationType.INSERTION:
            return self.mutate_insertion(individual)
        elif self.mutation_type == MutationType.INVERSION:
            return self.mutate_inversion(individual)
        elif self.mutation_type == MutationType.PERTURBATION:
            return self.mutate_perturbation(individual)
        return individual

    def select_survivors(self, offspring: List[List[int]]) -> None:
        """Seleciona sobreviventes substituindo os piores indivíduos."""
        all_individuals = self.population + offspring
        all_individuals.sort(key=self.calculate_fitness)
        self.population = all_individuals[:self.population_size]

    def run(self) -> Tuple[bool, int, List[float], List[float]]:
        """Executa o algoritmo genético."""
        self.initialize_population()
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluations = 0
        iteration = 0

        while self.evaluations < self.max_evaluations:
            # Calcula fitness da população atual
            population_fitness = [self.calculate_fitness(ind) for ind in self.population]
            self.evaluations += len(self.population)
            
            # Atualiza histórico
            self.fitness_history.append(np.mean(population_fitness))
            self.best_fitness_history.append(min(population_fitness))
            
            # Verifica condição de término
            if min(population_fitness) == 0:
                return True, iteration, self.fitness_history, self.best_fitness_history

            # Gera nova geração
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])

            self.select_survivors(offspring)
            iteration += 1

        return False, iteration, self.fitness_history, self.best_fitness_history

def run_experiment(mutation_type: MutationType, num_runs: int = 30) -> None:
    """Executa o experimento múltiplas vezes e coleta estatísticas."""
    converged_runs = 0
    iterations_to_converge = []
    final_fitness_values = []

    for run in range(num_runs):
        print(f"Executando experimento {run + 1}/{num_runs} com mutação {mutation_type.value}")
        ga = EightQueensGA(mutation_type)
        converged, iterations, _, _ = ga.run()
        
        if converged:
            converged_runs += 1
            iterations_to_converge.append(iterations)
        
        final_fitness = min([ga.calculate_fitness(ind) for ind in ga.population])
        final_fitness_values.append(final_fitness)

    # Análise dos resultados
    print(f"\nResultados do experimento com mutação {mutation_type.value}:")
    print(f"Número de execuções que convergiram: {converged_runs}/{num_runs}")
    if iterations_to_converge:
        print(f"Média de iterações para convergência: {np.mean(iterations_to_converge):.2f}")
        print(f"Desvio padrão das iterações: {np.std(iterations_to_converge):.2f}")
    print(f"Fitness médio final: {np.mean(final_fitness_values):.2f}")
    print(f"Desvio padrão do fitness final: {np.std(final_fitness_values):.2f}")

    # Plotando gráficos
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ga.fitness_history, label='Fitness Médio')
    plt.plot(ga.best_fitness_history, label='Melhor Fitness')
    plt.xlabel('Iteração')
    plt.ylabel('Fitness')
    plt.title(f'Convergência do Algoritmo - Mutação {mutation_type.value}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Executa experimentos com diferentes tipos de mutação
    for mutation_type in MutationType:
        run_experiment(mutation_type) 