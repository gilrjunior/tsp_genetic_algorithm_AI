from GeneticAlgorithm import GeneticAlgorithm
from interface import Interface
from mock_data import get_mock_data
from distances_map import get_distances_map

def main():

    genetic_algorithm = GeneticAlgorithm(population_size=3, mutation_rate=0.01, crossover_rate=0.9, elitism_count=10, selection_method='roulette', tournament_size=5)
    genetic_algorithm.initialize_population()
    
    for route in genetic_algorithm.current_population:
        print(route)

    print(genetic_algorithm.fitness())
    print(genetic_algorithm.best_individual)
    print(genetic_algorithm.best_fitness)

if __name__ == "__main__":
    main() 