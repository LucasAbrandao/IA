import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

POP_SIZE = 100
NUM_GENES = 2
BOUNDS = [-10, 10]
GENS = 100
CROSS_RATE = 0.8
MUT_RATE = 0.1

def alpine2(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

def init_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, NUM_GENES))

def evaluate(pop):
    return alpine2(pop)

def select(pop, fitness):
    idx = np.random.choice(len(pop), 3)
    best = idx[np.argmax(fitness[idx])]
    return pop[best]

def crossover(p1, p2):
    if random.random() < CROSS_RATE:
        point = random.randint(1, NUM_GENES-1)
        return np.concatenate((p1[:point], p2[point:]))
    return p1.copy()

def mutate(ind):
    for i in range(NUM_GENES):
        if random.random() < MUT_RATE:
            ind[i] += np.random.normal(0, 1)
            ind[i] = np.clip(ind[i], BOUNDS[0], BOUNDS[1])
    return ind

def gerar_curvas_nivel():
    x = np.linspace(BOUNDS[0], BOUNDS[1], 200)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.abs(X * np.sin(X) + 0.1 * X) + np.abs(Y * np.sin(Y) + 0.1 * Y)
    return X, Y, Z

def run_genetic():
    pop = init_population()
    geracoes = [pop.copy()]

    for _ in range(GENS):
        fitness = evaluate(pop)
        new_pop = []
        for _ in range(POP_SIZE):
            p1 = select(pop, fitness)
            p2 = select(pop, fitness)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        pop = np.array(new_pop)
        geracoes.append(pop.copy())
    return geracoes

def animar_populacao(geracoes):
    X, Y, Z = gerar_curvas_nivel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    pontos, = ax.plot([], [], 'ro', markersize=4)

    def init():
        pontos.set_data([], [])
        return pontos,

    def update(frame):
        pop = geracoes[frame]
        pontos.set_data(pop[:, 0], pop[:, 1])
        ax.set_title(f'Geração {frame}')
        return pontos,

    ani = animation.FuncAnimation(fig, update, frames=len(geracoes), init_func=init,
                                  blit=True, repeat=False)
    ani.save('alpine2_evolution.mp4', fps=10, dpi=150)
    plt.close()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    geracoes = run_genetic()
    animar_populacao(geracoes)
    print("Vídeo gerado: alpine2_evolution.mp4")

