from .GeneticAlgorithm import GeneticAlgorithm
from .interface import Interface
from .mock_data import get_mock_data
from .distances_map import get_distances_map

def print_generation_info(generation, best_individuals, best_fitnesses, global_best_individual, global_best_fitness):
    """Callback para imprimir informações sobre a geração atual"""
    print(f"\nGeração {generation}")
    print("Melhores fitness por população:")
    for i, fitness in enumerate(best_fitnesses):
        print(f"População {i+1}: {fitness:.2f}")
    print(f"Melhor fitness global: {global_best_fitness:.2f}")
    # print(f"Melhor rota global: {global_best_individual}")

def main():
    # Parâmetros do algoritmo
    # population_size = 100
    # mutation_rate = 0.2
    # crossover_rate = 0.85
    # elitism_count = 10
    # selection_method = 'roulette'
    # tournament_size = 3
    # num_populations = 3
    # migration_interval = 15
    # migration_count = 1
    # generations = 150

    # genetic_algorithm = GeneticAlgorithm(population_size=200, mutation_rate=0.2, crossover_rate=0.85, elitism_count=2, selection_method='roulette', tournament_size=5)
    
    # best_individual, best_fitness = genetic_algorithm.run(generations=100)

    # Cria e executa o algoritmo genético
    # ga = GeneticAlgorithm(
    #     population_size=population_size,
    #     mutation_rate=mutation_rate,
    #     crossover_rate=crossover_rate,
    #     elitism_count=elitism_count,
    #     selection_method=selection_method,
    #     tournament_size=tournament_size,
    #     num_populations=num_populations,
    #     migration_interval=migration_interval,
    #     migration_count=migration_count
    # )

    # Executa o algoritmo
    # best_individual, best_fitness = ga.run(generations, print_generation_info)

    # print(f"Melhor rota: {best_individual}")
    # print(f"Melhor fitness: {best_fitness}")
    # print(f"\nLink do Google Maps para a rota:")
    # print(best_individual.get_google_maps_url())

    # Imprime o resultado final
    # print("\nResultado Final:")
    # print(f"Melhor fitness encontrado: {best_fitness:.2f}")
    # print(f"Menor distância encontrada: {1000 - best_fitness:.2f}")
    # print(f"Melhor rota encontrada: {best_individual}")
    # print(f"Rota no google maps: {best_individual.get_google_maps_url()}")

    interface = Interface()
    interface.run()

if __name__ == "__main__":
    main() 