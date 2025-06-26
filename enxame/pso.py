import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função de Rastrigin
def rastrigin(position):
    x, y = position
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# Parâmetros do PSO
num_particles = 30
dimensions = 2
iterations = 100
x_min, x_max = -5.12, 5.12

w = 0.7    # peso de inércia
c1 = 1.5   # coeficiente cognitivo (pessoal)
c2 = 1.5   # coeficiente social (global)

# Inicialização
positions = np.random.uniform(x_min, x_max, (num_particles, dimensions))
velocities = np.zeros((num_particles, dimensions))

pbest_positions = positions.copy()
pbest_scores = np.array([rastrigin(p) for p in positions])
gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index].copy()

# Para visualização
positions_over_time = [positions.copy()]

# Loop de otimização
for t in range(iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - positions[i])
            + c2 * r2 * (gbest_position - positions[i])
        )
        positions[i] += velocities[i]

        # Atualizar pbest se necessário
        score = rastrigin(positions[i])
        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = positions[i].copy()

    # Atualizar gbest
    gbest_index = np.argmin(pbest_scores)
    gbest_position = pbest_positions[gbest_index].copy()

    positions_over_time.append(positions.copy())

# Visualização com gráfico de contorno
x = np.linspace(x_min, x_max, 200)
y = np.linspace(x_min, x_max, 200)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

fig, ax = plt.subplots()
ax.contourf(X, Y, Z, levels=50, cmap='viridis')
scat = ax.scatter([], [], c='red', s=30)

def update(frame):
    scat.set_offsets(positions_over_time[frame])
    ax.set_title(f"Iteração {frame}")
    return scat,

anim = FuncAnimation(fig, update, frames=len(positions_over_time), interval=100)
plt.show()
