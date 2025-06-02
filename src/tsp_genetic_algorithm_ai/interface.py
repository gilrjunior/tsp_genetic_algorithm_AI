import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import numpy as np
import webbrowser

from .GeneticAlgorithm import GeneticAlgorithm

class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Algoritmo Genético - TSP")
        self.root.geometry("1200x800")
        
        # Configuração do estilo
        self.style = ttk.Style()
        self.style.configure("Red.TButton", foreground="red")
        
        # Variáveis de controle
        self.is_running = False
        self.stop_flag = False
        self.entries = {}
        self.selection_method = tk.StringVar(value="roulette")
        self.tournament_size = tk.StringVar(value="3")
        
        # Cria os frames
        self.create_frames()
        
        # Cria os controles
        self.create_controls()
        
        # Cria os gráficos
        self.create_graphs()
        
        # Cria os labels de status
        self.create_status_labels()

    def create_frames(self):
        # Frame principal para controles e informações
        self.frame_left = ttk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Frame para controles
        self.frame_controls = ttk.LabelFrame(self.frame_left, text="Controles", padding=10)
        self.frame_controls.pack(fill=tk.X, pady=(0, 10))
        
        # Frame para status
        self.frame_status = ttk.LabelFrame(self.frame_left, text="Status", padding=10)
        self.frame_status.pack(fill=tk.X, pady=(0, 10))
        
        # Frame para informações detalhadas
        self.frame_details = ttk.LabelFrame(self.frame_left, text="Informações Detalhadas", padding=10)
        self.frame_details.pack(fill=tk.X)
        
        # Frame para gráficos
        self.frame_graph = ttk.LabelFrame(self.root, text="Gráficos", padding=10)
        self.frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_controls(self):
        # Parâmetros do AG
        labels_and_defaults = [
            ("Tamanho da População:", "100"),
            ("Probabilidade de Cruzamento:", "0.8"),
            ("Probabilidade de Mutação:", "0.1"),
            ("Número de Gerações:", "100"),
            ("Tamanho do Elitismo:", "2"),
            ("Número de Populações:", "3"),
            ("Intervalo de Migração:", "10"),
            ("Quantidade de Migrantes:", "1")
        ]
        
        for i, (label, default) in enumerate(labels_and_defaults):
            ttk.Label(self.frame_controls, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(self.frame_controls, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.entries[label] = entry
        
        # Método de seleção
        ttk.Label(self.frame_controls, text="Método de Seleção:").grid(row=len(labels_and_defaults), column=0, padx=5, pady=5, sticky="w")
        selection_frame = ttk.Frame(self.frame_controls)
        selection_frame.grid(row=len(labels_and_defaults), column=1, padx=5, pady=5, sticky="w")
        
        ttk.Radiobutton(selection_frame, text="Roleta", variable=self.selection_method, value="roulette").pack(side=tk.LEFT)
        ttk.Radiobutton(selection_frame, text="Torneio", variable=self.selection_method, value="tournament").pack(side=tk.LEFT)
        
        # Tamanho do torneio
        ttk.Label(self.frame_controls, text="Tamanho do Torneio:").grid(row=len(labels_and_defaults)+1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.frame_controls, textvariable=self.tournament_size, width=10).grid(row=len(labels_and_defaults)+1, column=1, padx=5, pady=5, sticky="w")
        
        # Botões
        btn_start = ttk.Button(
            self.frame_controls,
            text="Iniciar",
            command=self.start_algorithm
        )
        btn_start.grid(row=len(labels_and_defaults)+2, column=0, padx=5, pady=10, sticky="w")
        
        btn_stop = ttk.Button(
            self.frame_controls,
            text="Parar",
            style="Red.TButton",
            command=self.stop_algorithm
        )
        btn_stop.grid(row=len(labels_and_defaults)+2, column=1, padx=5, pady=10, sticky="w")

    def create_graphs(self):
        # Cria uma grade de subplots para os gráficos
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        # Gráfico de fitness por população
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title("Evolução do Fitness por População")
        self.ax1.set_xlabel("Geração")
        self.ax1.set_ylabel("Fitness")
        self.ax1.grid(True)
        
        # Gráfico do melhor global
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title("Melhor Fitness Global")
        self.ax2.set_xlabel("Geração")
        self.ax2.set_ylabel("Fitness")
        self.ax2.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graph)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializa as listas para armazenar os dados dos gráficos
        self.generations = []
        self.population_fitnesses = []
        self.global_fitnesses = []

    def create_status_labels(self):
        # Frame para informações básicas
        basic_info_frame = ttk.Frame(self.frame_status)
        basic_info_frame.pack(fill=tk.X, pady=5)
        
        self.generation_label = ttk.Label(basic_info_frame, text="Geração: 0")
        self.generation_label.pack(side=tk.LEFT, padx=10)
        
        self.best_fitness_label = ttk.Label(basic_info_frame, text="Melhor Fitness Global: 0.0")
        self.best_fitness_label.pack(side=tk.LEFT, padx=10)
        
        self.distance_label = ttk.Label(basic_info_frame, text="Distância Total: 0.0 km")
        self.distance_label.pack(side=tk.LEFT, padx=10)
        
        # Frame para a rota
        route_frame = ttk.Frame(self.frame_status)
        route_frame.pack(fill=tk.X, pady=5)
        
        self.best_route_label = ttk.Label(route_frame, text="Melhor Rota: ", wraplength=300)
        self.best_route_label.pack(side=tk.LEFT, padx=10)
        
        # Frame para fitness por população
        fitness_frame = ttk.Frame(self.frame_details)
        fitness_frame.pack(fill=tk.X, pady=5)
        
        self.population_fitness_labels = []
        for i in range(3):  # Assumindo 3 populações por padrão
            label = ttk.Label(fitness_frame, text=f"População {i+1} - Melhor Fitness: 0.0")
            label.pack(fill=tk.X, padx=10, pady=2)
            self.population_fitness_labels.append(label)

    def start_algorithm(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_flag = False
        
        # Limpa os gráficos
        self.generations = []
        self.population_fitnesses = []
        self.global_fitnesses = []
        self.ax1.clear()
        self.ax1.set_title("Evolução do Fitness por População")
        self.ax1.set_xlabel("Geração")
        self.ax1.set_ylabel("Fitness")
        self.ax1.grid(True)
        
        # Lê os parâmetros
        population_size = int(self.entries["Tamanho da População:"].get())
        crossover_prob = float(self.entries["Probabilidade de Cruzamento:"].get())
        mutation_prob = float(self.entries["Probabilidade de Mutação:"].get())
        generations = int(self.entries["Número de Gerações:"].get())
        elitism_size = int(self.entries["Tamanho do Elitismo:"].get())
        tournament_size = int(self.tournament_size.get())
        selection_method = self.selection_method.get()
        num_populations = int(self.entries["Número de Populações:"].get())
        migration_interval = int(self.entries["Intervalo de Migração:"].get())
        migration_count = int(self.entries["Quantidade de Migrantes:"].get())

        # Cria a instância do algoritmo genético
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            crossover_rate=crossover_prob,
            mutation_rate=mutation_prob,
            elitism_count=elitism_size,
            tournament_size=tournament_size,
            selection_method=selection_method,
            num_populations=num_populations,
            migration_interval=migration_interval,
            migration_count=migration_count
        )

        # Configura o callback de parada
        self.ga.stop = lambda: self.stop_flag

        # Inicia o algoritmo em uma thread separada
        self.algorithm_thread = threading.Thread(
            target=self.ga.run,
            args=(generations, self.update_display),
            daemon=True
        )
        self.algorithm_thread.start()

    def stop_algorithm(self):
        self.stop_flag = True
        self.is_running = False
        
        # Imprime a rota final no terminal
        if hasattr(self, 'current_route'):
            print("\n=== Rota Final ===")
            print(f"Rota: {self.current_route}")
            print(f"Distância Total: {self.current_distance:.2f} km")
            print(f"Link do Google Maps: {self.current_route.get_google_maps_url()}")
            print("=================\n")

    def update_display(self, generation, best_individuals, best_fitnesses, global_best_individual, global_best_fitness):
        """Atualiza a interface com os resultados do algoritmo"""
        self.generation_label.config(text=f"Geração: {generation}")
        self.best_fitness_label.config(text=f"Melhor Fitness Global: {global_best_fitness:.2f}")
        
        # Atualiza os fitness por população
        for i, fitness in enumerate(best_fitnesses):
            if i < len(self.population_fitness_labels):
                self.population_fitness_labels[i].config(text=f"População {i+1} - Melhor Fitness: {fitness:.2f}")
        
        # Atualiza a rota e distância
        if global_best_individual:
            # Atualiza a rota
            route_str = str(global_best_individual)
            self.best_route_label.config(text=f"Melhor Rota: {route_str}")
            
            # Calcula a distância total
            total_distance = 1000 - global_best_fitness
            
            self.distance_label.config(text=f"Distância Total: {total_distance:.2f} km")
            
            # Armazena a rota atual para impressão final
            self.current_route = global_best_individual
            self.current_distance = total_distance
        
        # Atualiza os dados dos gráficos
        self.generations.append(generation)
        self.population_fitnesses.append(best_fitnesses)
        self.global_fitnesses.append(global_best_fitness)
        
        # Atualiza o gráfico de fitness por população
        self.ax1.clear()
        self.ax1.set_title("Evolução do Fitness por População")
        self.ax1.set_xlabel("Geração")
        self.ax1.set_ylabel("Fitness")
        self.ax1.grid(True)
        
        for i in range(len(best_fitnesses)):
            population_fitness = [f[i] for f in self.population_fitnesses]
            self.ax1.plot(self.generations, population_fitness, label=f"População {i+1}")
        
        self.ax1.legend()
        
        # Atualiza o gráfico do melhor global
        self.ax2.clear()
        self.ax2.set_title("Melhor Fitness Global")
        self.ax2.set_xlabel("Geração")
        self.ax2.set_ylabel("Fitness")
        self.ax2.grid(True)
        self.ax2.plot(self.generations, self.global_fitnesses, 'r-', label="Melhor Global")
        self.ax2.legend()
        
        self.canvas.draw()
        
        # Atualiza a interface
        self.root.update_idletasks()

    def on_closing(self):
        self.stop_flag = True
        if hasattr(self, "algorithm_thread") and self.algorithm_thread.is_alive():
            self.algorithm_thread.join(timeout=2)
        self.root.destroy()
        os._exit(0)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()