import numpy as np
import matplotlib.pyplot as plt

# --- PARÂMETROS DO ACO ---
NUM_ANTS = 30
NUM_ITERATIONS = 500
ALPHA = 1.0
BETA = 5.0
RHO = 0.2
Q = 1.0
CONVERGENCE_THRESHOLD = 100

np.random.seed(42)  # Reprodutibilidade

# --- CARREGAMENTO DA MATRIZ DE DISTÂNCIAS ---
distances = np.loadtxt('distancia_matrix.csv', delimiter=',')
num_cities = len(distances)

# --- INICIALIZAÇÃO DO FEROMÔNIO ---
pheromone = np.ones((num_cities, num_cities))

# --- FUNÇÕES AUXILIARES ---
def build_route(pheromone, distances, alpha, beta):
    route = []
    visited = set()
    current_city = np.random.randint(num_cities)
    route.append(current_city)
    visited.add(current_city)

    while len(route) < num_cities:
        unvisited = [city for city in range(num_cities) if city not in visited]
        probs = []

        for city in unvisited:
            tau = pheromone[current_city][city] ** alpha
            eta = (1.0 / distances[current_city][city]) ** beta
            probs.append(tau * eta)

        probs = np.array(probs)
        probs /= probs.sum()

        next_city = np.random.choice(unvisited, p=probs)
        route.append(next_city)
        visited.add(next_city)
        current_city = next_city

    return route

def calculate_length(route, distances):
    return sum(distances[route[i]][route[(i+1)%num_cities]] for i in range(num_cities))

def update_pheromones(pheromone, routes, lengths, rho, Q):
    pheromone *= (1 - rho)
    for route, length in zip(routes, lengths):
        for i in range(num_cities):
            a, b = route[i], route[(i + 1) % num_cities]
            pheromone[a][b] += Q / length
            pheromone[b][a] = pheromone[a][b]

# --- EXECUÇÃO PRINCIPAL ---
best_route = None
best_length = float('inf')
best_lengths = []
stable_iterations = 0

for iteration in range(NUM_ITERATIONS):
    routes = []
    lengths = []

    for _ in range(NUM_ANTS):
        route = build_route(pheromone, distances, ALPHA, BETA)
        length = calculate_length(route, distances)
        routes.append(route)
        lengths.append(length)

    min_length = min(lengths)
    if min_length < best_length:
        best_length = min_length
        best_route = routes[lengths.index(min_length)]
        stable_iterations = 0
    else:
        stable_iterations += 1

    best_lengths.append(best_length)
    update_pheromones(pheromone, routes, lengths, RHO, Q)

    if stable_iterations >= CONVERGENCE_THRESHOLD:
        print(f"Convergência atingida na iteração {iteration}")
        break

# --- RESULTADOS ---
print("\n--- Resultados Finais ---")
print(f"Melhor rota: {best_route}")
print(f"Distância total: {best_length:.2f}")

# --- VISUALIZAÇÃO ---
plt.figure(figsize=(10, 5))
plt.plot(best_lengths, linewidth=2)
plt.xlabel("Iterações")
plt.ylabel("Melhor Distância")
plt.title("Evolução da Melhor Distância - ACO")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- SALVAMENTO DA ROTA ---
np.savetxt("melhor_rota.csv", best_route, fmt='%d', delimiter=",")
