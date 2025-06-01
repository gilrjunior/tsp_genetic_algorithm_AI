from GeneticAlgorithm import GeneticAlgorithm
from interface import Interface
from mock_data import get_mock_data
from distances_map import get_distances_map

def main():

    genetic_algorithm = GeneticAlgorithm(population_size=200, mutation_rate=0.2, crossover_rate=0.85, elitism_count=2, selection_method='roulette', tournament_size=5)
    
    best_individual, best_fitness = genetic_algorithm.run(generations=100)

    print(f"Melhor rota: {best_individual}")
    print(f"Melhor fitness: {best_fitness}")
    print(f"\nLink do Google Maps para a rota:")
    print(best_individual.get_google_maps_url())

if __name__ == "__main__":
    main() 