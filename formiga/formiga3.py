import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_aco_tsp(
    dist_csv='distancia_matrix.csv',
    num_ants=None,
    max_iters=1000,
    alpha=1.0,
    beta=5.0,
    rho=0.5,
    Q=10.0,
    seed=89
):
    # Reprodutibilidade
    np.random.seed(seed)
    
    # Carrega matriz de distâncias
    dist_df = pd.read_csv(dist_csv, index_col=0)
    D = dist_df.values
    n = D.shape[0]
    if num_ants is None:
        num_ants = n 

    # Inicializa feromônio e heurística (1/d)
    tau = np.ones((n, n))
    eta = 1 / (D + np.diag([np.inf]*n))
    
    best_dist = np.inf
    best_tour = None
    history = []

    for it in range(max_iters):
        ants_tours = []
        ants_len = []

        for _ in range(num_ants):
            start = np.random.randint(n)
            tour = [start]
            unvisited = set(range(n)) - {start}
            cur = start

            # Construção probabilística do caminho
            while unvisited:
                weights = np.array([
                    (tau[cur, j]*alpha) * (eta[cur, j]*beta)
                    for j in unvisited
                ])
                probs = weights / weights.sum()
                nxt = np.random.choice(list(unvisited), p=probs)
                tour.append(nxt)
                unvisited.remove(nxt)
                cur = nxt

            tour.append(start)  # volta à origem
            L = sum(D[tour[i], tour[i+1]] for i in range(len(tour)-1))
            ants_tours.append(tour)
            ants_len.append(L)

        # atualiza melhor global
        idx = np.argmin(ants_len)
        if ants_len[idx] < best_dist:
            best_dist = ants_len[idx]
            best_tour = ants_tours[idx]
        history.append(best_dist)

        # evaporação
        tau *= (1 - rho)
        # depósito
        for tour, L in zip(ants_tours, ants_len):
            deposit = Q / L
            for i in range(len(tour)-1):
                a, b = tour[i], tour[i+1]
                tau[a, b] += deposit
                tau[b, a] += deposit

    return best_dist, best_tour, history

if __name__ == '__main__':
    best_dist, best_tour, hist = run_aco_tsp(
        dist_csv='distancia_matrix.csv',
        max_iters=1000,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=1.0,
        seed=123
    )

    print(f'Melhor distância: {best_dist}')
    print(f'Melhor tour: {best_tour}')

    # Gráfico de convergência
    plt.figure(figsize=(8,5))
    plt.plot(hist, lw=1)
    plt.xlabel('Iteração')
    plt.ylabel('Melhor distância até agora')
    plt.title('Convergência ACO - Caxeiro Viajante 20 Cidades')
    plt.grid(True)
    plt.tight_layout()
    plt.show()