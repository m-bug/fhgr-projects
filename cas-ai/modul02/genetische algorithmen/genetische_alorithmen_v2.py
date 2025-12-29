import random
import math
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Problemdefinition
# =========================================================
X_MIN = 0.0
X_MAX = 10.0

POP_SIZE = 40
GENERATIONS = 45
MUTATION_RATE = 0.1

def fitness(x):
    return x * math.sin(x)

# =========================================================
# GRAFIK 1 – DAS PROBLEM (Zielfunktion)
# =========================================================
x_vals = np.linspace(X_MIN, X_MAX, 400)
y_vals = [fitness(x) for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("G1: Zielfunktion mit mehreren lokalen Maxima")

plt.grid(False)
plt.tick_params(axis="both", which="both", length=0)

plt.savefig("slide_1_zielfunktion.png", dpi=250, bbox_inches="tight")

plt.show()

# =========================================================
# GENETISCHER ALGORITHMUS
# =========================================================
population = [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]
history = []

for _ in range(GENERATIONS):
    history.append(population.copy())

    population.sort(key=fitness, reverse=True)
    survivors = population[:POP_SIZE // 2]

    new_population = survivors.copy()

    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(survivors, 2)
        child = (p1 + p2) / 2.0

        if random.random() < MUTATION_RATE:
            child += random.uniform(-0.5, 0.5)

        child = max(X_MIN, min(X_MAX, child))
        new_population.append(child)

    population = new_population

# =========================================================
# GRAFIK 2 – DER LÖSUNGSPROZESS (Population)
# =========================================================
plt.figure(figsize=(10, 5))

for gen, pop in enumerate(history):
    plt.scatter(
        pop,
        [gen] * len(pop),
        s=10,
        alpha=0.6
    )

plt.xlabel("x (Lösungsraum)")
plt.ylabel("Generation")
plt.title("G2: Evolution der Population im genetischen Algorithmus")

plt.grid(False)
plt.tick_params(axis="both", which="both", length=0)

plt.savefig("slide_2_population.png", dpi=250, bbox_inches="tight")

plt.show()
