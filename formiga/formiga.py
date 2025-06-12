# Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CARREGAMENTO DA MATRIZ DE DISTÂNCIAS ---
# Lê o arquivo CSV contendo as distâncias entre as cidades
distances = np.loadtxt('distancia_matrix.csv', delimiter=',')
num_cities = len(distances)  # Número de cidades (20 no exemplo)
print(f"Matriz de distâncias carregada. Shape: {distances.shape}")

# --- 2. PARÂMETROS DO ALGORITMO ACO ---
num_ants = 10          # Número de formigas
num_iterations = 200   # Número máximo de iterações
alpha = 1.0            # Influência do feromônio na escolha do caminho
beta = 2.0             # Influência da distância (heurística) na escolha
rho = 0.5              # Taxa de evaporação do feromônio (0 < rho <= 1)
Q = 1.0                # Quantidade de feromônio depositado por formiga

# Inicialização da matriz de feromônios (todas as arestas começam com valor 1)
pheromone = np.ones((num_cities, num_cities))

# --- 3. FUNÇÃO PARA CONSTRUIR UMA ROTA (ANT SOLUTION CONSTRUCTION) ---
def build_route(pheromone, distances, alpha, beta):
    route = []
    visited = set()
    current_city = np.random.randint(num_cities)  # Escolhe uma cidade inicial aleatória
    route.append(current_city)
    visited.add(current_city)

    # Constrói a rota até visitar todas as cidades
    while len(route) < num_cities:
        unvisited = [city for city in range(num_cities) if city not in visited]
        probabilities = []

        # Calcula probabilidades para cidades não visitadas
        for city in unvisited:
            pheromone_val = pheromone[current_city][city] ** alpha
            heuristic_val = (1.0 / distances[current_city][city]) ** beta  # Inverso da distância
            probabilities.append(pheromone_val * heuristic_val)

        # Normaliza as probabilidades para soma = 1
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        # Escolhe a próxima cidade probabilisticamente (roleta viciada)
        next_city = np.random.choice(unvisited, p=probabilities)
        route.append(next_city)
        visited.add(next_city)
        current_city = next_city

    return route

# --- 4. FUNÇÃO PARA CALCULAR O COMPRIMENTO DE UMA ROTA ---
def calculate_route_length(route, distances):
    length = 0
    for i in range(len(route)):
        city_a = route[i]
        city_b = route[(i + 1) % len(route)]  # Volta à primeira cidade no final
        length += distances[city_a][city_b]
    return length

# --- 5. ATUALIZAÇÃO DA MATRIZ DE FEROMÔNIOS (PHEROMONE UPDATE) ---
def update_pheromone(pheromone, routes, route_lengths, rho, Q):
    # Evaporação: reduz o feromônio em todas as arestas
    pheromone *= (1 - rho)

    # Depósito de feromônio pelas formigas
    for ant_idx in range(len(routes)):
        route = routes[ant_idx]
        length = route_lengths[ant_idx]
        for i in range(len(route)):
            city_a = route[i]
            city_b = route[(i + 1) % len(route)]
            pheromone[city_a][city_b] += Q / length  # Depósito proporcional à qualidade da rota
            pheromone[city_b][city_a] = pheromone[city_a][city_b]  # Simetria

# --- 6. EXECUÇÃO PRINCIPAL DO ALGORITMO ACO ---
best_route = None
best_length = float('inf')
best_lengths = []  # Armazena a melhor distância de cada iteração

for iteration in range(num_iterations):
    routes = []
    route_lengths = []

    # Cada formiga constrói uma rota
    for ant in range(num_ants):
        route = build_route(pheromone, distances, alpha, beta)
        length = calculate_route_length(route, distances)
        routes.append(route)
        route_lengths.append(length)

        # Atualiza a melhor solução global
        if length < best_length:
            best_length = length
            best_route = route

    # Atualiza os feromônios após todas as formigas terminarem
    update_pheromone(pheromone, routes, route_lengths, rho, Q)

    # Armazena a melhor distância desta iteração
    best_lengths.append(best_length)

    # Critério de parada: convergência (sem melhora por 100 iterações)
    if iteration >= 100 and len(set(best_lengths[-100:])) == 1:
        print(f"Convergência alcançada na iteração {iteration}!")
        break

# --- 7. RESULTADOS ---
print("\n--- Resultados Finais ---")
print(f"Melhor rota encontrada: {best_route}")
print(f"Melhor distância: {best_length}")

# --- 8. VISUALIZAÇÃO DA EVOLUÇÃO ---
plt.plot(best_lengths, linewidth=2)
plt.xlabel("Iteração", fontsize=12)
plt.ylabel("Melhor Distância", fontsize=12)
plt.title("Evolução da Melhor Distância no ACO", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()