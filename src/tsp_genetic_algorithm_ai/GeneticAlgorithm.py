import numpy as np
import math
import random
from distances_map import get_distances_map
from mock_data import get_mock_data
from Route import Route

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count = None, selection_method='roulette', tournament_size=None):
        """
        Inicializa os parâmetros do algoritmo genético.

        :param population_size: Tamanho da população.
        :param mutation_rate: Taxa de mutação.
        :param crossover_rate: Taxa de cruzamento.
        :param elitism_count: Número de indivíduos a serem selecionados para a próxima geração.
        :param selection_method: Método de seleção (roulette ou tournament).
        :param tournament_size: Tamanho do torneio (se selection_method for tournament).
        :param crossover_type: Tipo de cruzamento (single_point ou double_point).
        :param max_known_value: Valor máximo conhecido da função, nem sempre é conhecido.
        """

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.best_individual = None
        self.best_fitness = None
        self.current_population = None
        self.stop = None # Callback para parar o algoritmo

    def maximum_route_distance_function(self):
        """
        Função que retorna a distância máxima da rota.
        """

        fitness_values = np.array([])
        for route in self.current_population:
            distance = 0
            for i in range(len(route.locations)-1):
                distance += get_distances_map(route.locations[i].name, route.locations[i+1].name)
            fitness_values = np.append(fitness_values, distance)

        fitness_values = 1000 - fitness_values

        return fitness_values


    def initialize_population(self):
        """
        Cria a população inicial de indivíduos.
        """
        locations = get_mock_data()
        self.current_population = []

        for _ in range(self.population_size):
            # Cria uma cópia das locations exceto a primeira
            remaining_locations = locations[1:].copy()
            # Embaralha as locations restantes
            random.shuffle(remaining_locations)
            # Cria a rota com a primeira location no início e fim            
            route = [locations[0]] + remaining_locations + [locations[0]]
            self.current_population.append(Route(route))
    
    def fitness(self):
        """
        Calcula a aptidão (fitness) da população.
        Atualiza o melhor indivíduo, o erro do melhor e o erro médio da população.
        """
        fitness_values = self.maximum_route_distance_function()
        # Índice do melhor indivíduo
        best_idx = np.argmax(fitness_values)
        self.best_individual = self.current_population[best_idx]
        self.best_fitness = fitness_values[best_idx]
        
        return fitness_values

    def selection(self, fitness_values):
        """
        Seleciona os indivíduos para reprodução, com base no método definido.
        """
        # TODO implementar o elitismo

        # Seleciona o método de seleção
        if self.selection_method == 'roulette':
            # Seleciona os indivíduos para reprodução
            self.current_population = self.roulette_selection(fitness_values)
        elif self.selection_method == 'tournament':
            # Seleciona os indivíduos para reprodução
            self.current_population = self.tournament_selection(fitness_values)

    def roulette_selection(self, fitness_values):
        """
        Implementa a seleção por roleta.
        """

        # Calcula a probabilidade de cada indivíduo
        probabilities = fitness_values / np.sum(fitness_values)
        # Seleciona os indivíduos para reprodução
        selected_individuals = np.random.choice(len(self.current_population), size=self.population_size, p=probabilities)
        # Retorna os indivíduos selecionados
        return [self.current_population[i] for i in selected_individuals]

    def tournament_selection(self, fitness_values):
        """
        Implementa a seleção por torneio.
        """
        selected = []

        for _ in range(self.population_size):
            # Sorteia os indivíduos aleatórios da população de acordo com o tamanho do torneio
            participants_indices = np.random.choice(len(self.current_population), self.tournament_size, replace=False)
            participants_fitness = fitness_values[participants_indices]
            
            # Seleciona o melhor
            winner_indice = participants_indices[np.argmax(participants_fitness)]
            selected.append(self.current_population[winner_indice])

        return np.array(selected)

    def crossover(self):
        """
        Realiza o cruzamento entre dois pais (one-point ou two-point).
        """
        np.random.shuffle(self.current_population)
        children = []
        n = len(self.current_population)
        i = 0
        while i < n - 1: # Parar antes do último se for ímpar
            
            parent1 = self.current_population[i].locations
            parent2 = self.current_population[i+1].locations

            if random.random() < self.crossover_rate:

                swap_locations = self.cycle_crossover(parent1, parent2)

                child1 = parent1.copy()
                child2 = parent2.copy()

                for idx in swap_locations:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]

                children.extend([Route(child1), Route(child2)])

            else:
                children.extend([Route(parent1), Route(parent2)])
            i += 2

        self.current_population = np.array(children)

        if n % 2 == 1:
            children.append(self.current_population[-1])
        self.current_population = np.array(children)

    def cycle_crossover(self, parent1, parent2):
            
            """
            Realiza o crossover por ciclo.
            """

            # Pega os ids das locations
            parent1_locations_ids = [location.id for location in parent1]
            parent2_locations_ids = [location.id for location in parent2]
            
            # Remove primeiro e último elemento (que são iguais)
            parent1_locations_ids = parent1_locations_ids[1:-1]
            parent2_locations_ids = parent2_locations_ids[1:-1]
            
            # Define o array de indices de troca como uma lista
            swap_locations = list(range(len(parent1)))
            # Remove a fazenda, pois ela é a primeira e a última location
            swap_locations.remove(0)
            swap_locations.remove(21)
            
            # Remove o primeiro elemento, onde é que se inicia a rota
            swap_locations.remove(1)

            # Define o último id da troca
            last_location_id = parent2_locations_ids[0]

            while last_location_id != parent1_locations_ids[0]:
                index = parent1_locations_ids.index(last_location_id)
                swap_locations.remove(index+1)
                last_location_id = parent2_locations_ids[index]

            return swap_locations

    def mutation(self):
        """
        Aplica a mutação no indivíduo.
        """
        for route in self.current_population:
            if random.random() < self.mutation_rate:
                # Seleciona um indivíduo aleatório da população
                positions = random.sample(range(1, len(route.locations)-1), 2)

                #Troca as posições
                route.locations[positions[0]], route.locations[positions[1]] = route.locations[positions[1]], route.locations[positions[0]]
    
    def migration(self):
        """
        Aplica a migração entre as populações.
        """
        pass

    def run(self, generations, update_callback=None):
        """
        Executa o algoritmo genético por um número definido de gerações.
        
        :param generations: Número de gerações a serem executadas.
        :return: O melhor indivíduo encontrado.
        """
        elite_individuals = None
        self.initialize_population()  # Inicializa a população sem atribuir o retorno
        for generation in range(generations):

            if self.stop and self.stop():
                break

            print(f"Geração {generation + 1}")
            
            # Calcula a aptidão de cada indivíduo, 
            fitness_values = self.fitness()
            # Elitismo: mantém os melhores indivíduos da geração anterior
            if self.elitism_count and self.elitism_count > 0:
                elite_indices = np.argsort(fitness_values)[-self.elitism_count:].tolist()
                elite_individuals = [self.current_population[i] for i in elite_indices]

            # Faz a seleção, crossover e mutação
            self.selection(fitness_values)
            self.crossover() 
            self.mutation()

            if elite_individuals is not None:
                new_fitness_values = self.fitness()
                worst_indices = np.argsort(new_fitness_values)[:self.elitism_count]
                for i, idx in enumerate(worst_indices):
                    self.current_population[idx] = elite_individuals[i]

            self.fitness()

            if update_callback:
                update_callback(
                    generation= generation + 1,
                    best_individual=self.best_individual,
                    best_fitness=self.best_fitness,
                    error=self.current_error if self.current_error is not None else 0
                )

        return self.best_individual, self.best_fitness