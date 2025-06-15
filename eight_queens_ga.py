import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
from enum import Enum
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

class MutationType(Enum):
    SWAP = "swap"  # Troca de dois genes
    INSERTION = "insertion"  # Inserção de um gene próximo a outro
    INVERSION = "inversion"  # Inversão de uma sub-string
    PERTURBATION = "perturbation"  # Rearranjo aleatório de um subconjunto

@dataclass
class ExperimentResults:
    mutation_type: MutationType
    converged_runs: int
    iterations_to_converge: List[int]
    final_fitness_values: List[float]
    fitness_history: List[float]
    best_fitness_history: List[float]
    converged_individuals_per_run: List[int]

class ComparativeAnalysis:
    def __init__(self):
        self.results: Dict[MutationType, ExperimentResults] = {}
        self.percentile_points = 100  # Número de pontos percentuais para análise
        
    def add_results(self, mutation_type: MutationType, results: ExperimentResults):
        self.results[mutation_type] = results
    
    def normalize_progress(self, fitness_history: List[float], max_iterations: int) -> Tuple[List[float], List[float]]:
        """
        Normaliza o progresso do fitness para uma escala percentual.
        Retorna os percentuais e os valores de fitness normalizados.
        """
        # Cria pontos percentuais (0% a 100%)
        percentiles = np.linspace(0, 100, self.percentile_points)
        
        # Calcula os índices correspondentes para cada percentual
        indices = np.round(np.array(percentiles) * (len(fitness_history) - 1) / 100).astype(int)
        
        # Obtém os valores de fitness nos pontos percentuais
        normalized_fitness = [fitness_history[i] for i in indices]
        
        return percentiles, normalized_fitness

    def plot_convergence_comparison(self):
        """Plota a comparação da convergência entre diferentes tipos de mutação."""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Fitness médio por iteração
        plt.subplot(2, 1, 1)
        for mutation_type, result in self.results.items():
            # Usa o histórico de fitness da primeira execução
            iterations = range(len(result.fitness_history))
            plt.plot(iterations, result.fitness_history, 
                    label=mutation_type.value, 
                    linestyle='-', 
                    alpha=0.7)
        
        plt.xlabel('Iteração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness por Tipo de Mutação')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Configura os ticks do eixo x para mostrar apenas números inteiros
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Configura os ticks do eixo y para mostrar apenas números inteiros
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Subplot 2: Fitness médio normalizado por progresso percentual
        plt.subplot(2, 1, 2)
        for mutation_type, result in self.results.items():
            # Usa o tamanho do histórico de fitness como número máximo de iterações
            max_iterations = len(result.fitness_history)
            
            # Normaliza o progresso
            percentiles, normalized_fitness = self.normalize_progress(result.fitness_history, max_iterations)
            
            plt.plot(percentiles, normalized_fitness, 
                    label=mutation_type.value, 
                    linestyle='-', 
                    alpha=0.7)
        
        plt.xlabel('Progresso (%)')
        plt.ylabel('Fitness Médio')
        plt.title('Comparação do Fitness Médio Normalizado')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Configura os ticks do eixo x para mostrar apenas números inteiros
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # Configura os ticks do eixo y para mostrar apenas números inteiros
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig('convergence_analysis.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def generate_statistical_report(self):
        """Gera um relatório estatístico comparativo."""
        report = []
        for mutation_type, result in self.results.items():
            # Filtra apenas as execuções que convergiram
            converged_iterations = [it for it in result.iterations_to_converge if it > 0]
            
            stats = {
                'Tipo de Mutação': mutation_type.value,
                'Taxa de Convergência': f"{result.converged_runs}/30 ({result.converged_runs/30*100:.1f}%)",
                'Média de Iterações para Convergência': f"{np.mean(converged_iterations):.2f}" if converged_iterations else "N/A",
                'Desvio Padrão das Iterações': f"{np.std(converged_iterations):.2f}" if converged_iterations else "N/A",
                'Fitness Médio Final': f"{np.mean(result.final_fitness_values):.2f}",
                'Desvio Padrão do Fitness Final': f"{np.std(result.final_fitness_values):.2f}",
                'Média de Indivíduos Convergidos por Execução': f"{np.mean(result.converged_individuals_per_run):.2f}",
                'Iteração Média para Primeira Convergência': f"{np.mean([i for i, x in enumerate(result.best_fitness_history) if x == 0]):.2f}"
            }
            report.append(stats)
        
        # Cria um DataFrame para melhor visualização
        df = pd.DataFrame(report)
        print("\nRelatório Estatístico Comparativo:")
        print(df.to_string(index=False))
        
        # Salva o relatório em CSV
        df.to_csv('comparative_analysis.csv', index=False)
        print("\nRelatório salvo em 'comparative_analysis.csv'")

class EightQueensGA:
    def __init__(self, mutation_type: MutationType = MutationType.SWAP):
        self.population_size = 10
        self.crossover_prob = 0.9
        self.mutation_prob = 0.4
        self.max_evaluations = 10000
        self.min_iterations = 20  # Número mínimo de iterações
        self.board_size = 8
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluations = 0
        self.mutation_type = mutation_type
        self.found_solution = False  # Flag para indicar se encontrou a solução

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
        self.found_solution = False
        iteration = 0

        while self.evaluations < self.max_evaluations:
            # Calcula fitness da população atual
            population_fitness = [self.calculate_fitness(ind) for ind in self.population]
            self.evaluations += len(self.population)
            
            # Atualiza histórico
            self.fitness_history.append(np.mean(population_fitness))
            self.best_fitness_history.append(min(population_fitness))
            
            # Verifica se encontrou a solução
            if min(population_fitness) == 0:
                self.found_solution = True
            
            # Continua executando até atingir o número mínimo de iterações
            if self.found_solution and iteration >= self.min_iterations:
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

        return self.found_solution, iteration, self.fitness_history, self.best_fitness_history

def analyze_fastest_runs(results: ExperimentResults, num_runs: int = 5) -> None:
    """Analisa as execuções mais rápidas que convergiram."""
    # Filtra apenas as execuções que convergiram
    converged_iterations = [(i, it) for i, it in enumerate(results.iterations_to_converge) if it > 0]
    
    if not converged_iterations:
        print(f"\nNenhuma execução convergiu para o tipo de mutação {results.mutation_type.value}")
        return
    
    # Ordena por número de iterações
    converged_iterations.sort(key=lambda x: x[1])
    
    print(f"\nAnálise das {min(num_runs, len(converged_iterations))} execuções mais rápidas para {results.mutation_type.value}:")
    print("-" * 50)
    
    for i, (run_idx, iterations) in enumerate(converged_iterations[:num_runs]):
        print(f"\nExecução #{run_idx + 1}:")
        print(f"Iterações até convergência: {iterations}")
        print(f"Fitness final: {results.final_fitness_values[run_idx]}")
        print(f"Indivíduos convergidos: {results.converged_individuals_per_run[run_idx]}")

def run_experiment(mutation_type: MutationType, num_runs: int = 30) -> ExperimentResults:
    """Executa o experimento múltiplas vezes e coleta estatísticas detalhadas."""
    converged_runs = 0
    iterations_to_converge = []
    final_fitness_values = []
    converged_individuals_per_run = []
    best_fitness_history = []
    fitness_history = []
    detailed_runs = []  # Armazena detalhes de cada execução

    for run in range(num_runs):
        print(f"Executando experimento {run + 1}/{num_runs} com mutação {mutation_type.value}")
        ga = EightQueensGA(mutation_type)
        converged, iterations, run_fitness_history, run_best_fitness = ga.run()
        
        run_details = {
            'converged': converged,
            'iterations': iterations,
            'fitness_history': run_fitness_history,
            'best_fitness_history': run_best_fitness,
            'final_population_fitness': [ga.calculate_fitness(ind) for ind in ga.population]
        }
        detailed_runs.append(run_details)
        
        if converged:
            converged_runs += 1
            # Encontra a primeira iteração onde o fitness chegou a zero
            first_convergence = next((i for i, x in enumerate(run_best_fitness) if x == 0), -1)
            iterations_to_converge.append(first_convergence)
        else:
            iterations_to_converge.append(-1)  # Marca execuções não convergentes
        
        # Conta quantos indivíduos convergiram nesta execução
        converged_individuals = sum(1 for ind in ga.population if ga.calculate_fitness(ind) == 0)
        converged_individuals_per_run.append(converged_individuals)
        
        final_fitness = min([ga.calculate_fitness(ind) for ind in ga.population])
        final_fitness_values.append(final_fitness)
        
        # Atualiza históricos
        if not fitness_history:
            fitness_history = run_fitness_history
            best_fitness_history = run_best_fitness
        else:
            # Média dos históricos de todas as execuções
            fitness_history = [sum(x)/num_runs for x in zip(fitness_history, run_fitness_history)]
            best_fitness_history = [min(x) for x in zip(best_fitness_history, run_best_fitness)]

    results = ExperimentResults(
        mutation_type=mutation_type,
        converged_runs=converged_runs,
        iterations_to_converge=iterations_to_converge,
        final_fitness_values=final_fitness_values,
        fitness_history=fitness_history,
        best_fitness_history=best_fitness_history,
        converged_individuals_per_run=converged_individuals_per_run
    )
    
    # Analisa as execuções mais rápidas
    analyze_fastest_runs(results)
    
    return results

if __name__ == "__main__":
    # Cria o objeto de análise comparativa
    analysis = ComparativeAnalysis()
    
    # Executa experimentos para cada tipo de mutação
    for mutation_type in MutationType:
        results = run_experiment(mutation_type)
        analysis.add_results(mutation_type, results)
    
    # Gera relatórios e gráficos comparativos
    analysis.plot_convergence_comparison()
    analysis.generate_statistical_report() 