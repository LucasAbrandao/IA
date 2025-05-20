import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# ==== PARÂMETROS ====
POP_SIZE = 100
NUM_GENES = 2
BOUNDS = [-100, 100]
GENS = 100
CROSS_RATE = 0.8
MUT_RATE = 0.1

# ==== FUNÇÃO SCHAFFER F6 ====
def schaffer_f6(x):
    x1, x2 = x[:, 0], x[:, 1]
    num = np.square(np.sin(np.sqrt(x1**2 + x2**2))) - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 - (num / den)

# ==== INICIALIZAÇÃO ====
def init_population():
    return np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, NUM_GENES))

def evaluate(pop):
    return schaffer_f6(pop)

# ==== OPERAÇÕES GENÉTICAS ====
def select(pop, fitness):
    idx = np.random.choice(len(pop), 3)
    best = idx[np.argmax(fitness[idx])]
    return pop[best]

def crossover(p1, p2):
    if random.random() < CROSS_RATE:
        point = random.randint(1, NUM_GENES - 1)
        return np.concatenate((p1[:point], p2[point:]))
    return p1.copy()

def mutate_original(ind):
    for i in range(NUM_GENES):
        if random.random() < MUT_RATE:
            ind[i] += np.random.normal(0, 5)
            ind[i] = np.clip(ind[i], BOUNDS[0], BOUNDS[1])
    return ind

def mutate_adaptativa(ind, gen_ratio):
    for i in range(NUM_GENES):
        if random.random() < MUT_RATE:
            escala = (1 - gen_ratio) * (BOUNDS[1] - BOUNDS[0]) * 0.05
            ind[i] += np.random.normal(0, escala)
            ind[i] = np.clip(ind[i], BOUNDS[0], BOUNDS[1])
    return ind

# ==== CURVAS DE NÍVEL ====
def gerar_curvas_nivel():
    x = np.linspace(BOUNDS[0], BOUNDS[1], 300)
    y = np.linspace(BOUNDS[0], BOUNDS[1], 300)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = 0.5 - ((np.sin(R)**2 - 0.5) / (1 + 0.001 * R**2)**2)
    return X, Y, Z

# ==== ALGORITMOS GENÉTICOS ====
def run_genetic_basico():
    pop = init_population()
    geracoes = [pop.copy()]
    for _ in range(GENS):
        fitness = evaluate(pop)
        nova_pop = []
        for _ in range(POP_SIZE):
            p1, p2 = select(pop, fitness), select(pop, fitness)
            child = crossover(p1, p2)
            child = mutate_original(child)
            nova_pop.append(child)
        pop = np.array(nova_pop)
        geracoes.append(pop.copy())
    return geracoes

def run_genetic_melhorado():
    pop = init_population()
    geracoes = [pop.copy()]
    for gen in range(GENS):
        fitness = evaluate(pop)
        nova_pop = []
        for _ in range(POP_SIZE):
            p1, p2 = select(pop, fitness), select(pop, fitness)
            child = crossover(p1, p2)
            child = mutate_adaptativa(child, gen / GENS)
            nova_pop.append(child)
        pop = np.array(nova_pop)
        geracoes.append(pop.copy())
    return geracoes

# ==== ANIMAÇÃO ====
def animar_populacao(geracoes, nome_arquivo):
    X, Y, Z = gerar_curvas_nivel()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    pontos, = ax.plot([], [], 'ro', markersize=3)

    def init():
        pontos.set_data([], [])
        return pontos,

    def update(frame):
        pop = geracoes[frame]
        pontos.set_data(pop[:, 0], pop[:, 1])
        best = np.max(evaluate(pop))
        ax.set_title(f'Geração {frame} | Melhor fitness: {best:.4f}')
        return pontos,

    ani = animation.FuncAnimation(fig, update, frames=len(geracoes), init_func=init,
                                  blit=True, repeat=False)
    ani.save(nome_arquivo, fps=10)
    plt.close()

# ==== EXECUÇÃO PRINCIPAL ====
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("Gerando vídeo com mutação original...")
    ger_orig = run_genetic_basico()
    animar_populacao(ger_orig, "schaffer_original.gif")

    print("Gerando vídeo com mutação melhorada...")
    ger_melhorado = run_genetic_melhorado()
    animar_populacao(ger_melhorado, "schaffer_melhorado.gif")

    print("Pronto: vídeos gerados → schaffer_original.gif e schaffer_melhorado.gif")

