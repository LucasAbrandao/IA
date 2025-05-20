import random
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

NUM_ITENS = 30
PESO_MAX = 50

# Semente fixa
random.seed(42)
np.random.seed(42)

def gerar_itens():
    itens = [(random.randint(1, 20), random.randint(10, 100)) for _ in range(NUM_ITENS)]
    print("Itens gerados:")
    for i, (peso, valor) in enumerate(itens):
        print(f"Item {i}: Peso = {peso}, Valor = {valor}")
    return itens

def forca_bruta(itens, capacidade):
    melhor_valor = 0
    melhor_comb = []
    for i in range(len(itens)+1):
        for comb in combinations(itens, i):
            peso = sum(x[0] for x in comb)
            valor = sum(x[1] for x in comb)
            if peso <= capacidade and valor > melhor_valor:
                melhor_valor = valor
                melhor_comb = comb
    return melhor_comb, melhor_valor

def inicializar_pop(tam_pop, num_genes):
    return np.random.choice([0, 1], size=(tam_pop, num_genes), p=[0.8, 0.2])

def avaliar(individuo, itens):
    peso_total = sum(g * p for g, (p, _) in zip(individuo, itens))
    valor_total = sum(g * v for g, (_, v) in zip(individuo, itens))
    if peso_total <= PESO_MAX:
        return valor_total
    else:
        return valor_total - 5 * (peso_total - PESO_MAX)  # penalização suave

def selecao(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k)
    melhor = idx[np.argmax([fitness[i] for i in idx])]
    return pop[melhor].copy()

def crossover(p1, p2):
    if random.random() < 0.8:
        ponto = random.randint(1, len(p1)-1)
        return np.concatenate((p1[:ponto], p2[ponto:]))
    return p1.copy()

def mutacao(ind, taxa=0.05):
    for i in range(len(ind)):
        if random.random() < taxa:
            ind[i] = 1 - ind[i]
    return ind

def algoritmo_genetico(itens, geracoes=100, tam_pop=100):
    pop = inicializar_pop(tam_pop, len(itens))
    historico = []

    for g in range(geracoes):
        fitness = [avaliar(ind, itens) for ind in pop]
        nova_pop = []

        for _ in range(tam_pop):
            pai1 = selecao(pop, fitness)
            pai2 = selecao(pop, fitness)
            filho = crossover(pai1, pai2)
            filho = mutacao(filho)
            nova_pop.append(filho)

        pop = np.array(nova_pop)
        fitness = [avaliar(ind, itens) for ind in pop]
        historico.append((max(fitness), sum(fitness)/len(fitness), min(fitness)))

    melhor_ind = pop[np.argmax(fitness)]
    return melhor_ind, avaliar(melhor_ind, itens), historico

def plotar(historico):
    melhores = [h[0] for h in historico]
    medios = [h[1] for h in historico]
    piores = [h[2] for h in historico]

    plt.plot(melhores, label='Melhor')
    plt.plot(medios, label='Média')
    plt.plot(piores, label='Pior')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title("Evolução do Fitness")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    itens = gerar_itens()

    melhor_comb, val_forca = forca_bruta(itens[:15], PESO_MAX)
    print("Força Bruta (15 itens):", val_forca)

    melhor_ind, val_ag, historico = algoritmo_genetico(itens)
    print("AG (30 itens):", val_ag)
    print("Itens Selecionados:", [i for i, b in enumerate(melhor_ind) if b])

    plotar(historico)
