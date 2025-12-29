import random
import matplotlib.pyplot as plt

# -----------------------------
# Parameter
# -----------------------------
TARGET = 42
POP_SIZE = 30
GENERATIONS = 40
MUTATION_RATE = 0.1
VALUE_RANGE = (0, 100)

# -----------------------------
# Fitness-Funktion
# -----------------------------
def fitness(x):
    return -abs(TARGET - x)

# -----------------------------
# Initiale Population
# -----------------------------
population = [
    random.randint(VALUE_RANGE[0], VALUE_RANGE[1])
    for _ in range(POP_SIZE)
]

# Historie für Visualisierung
history = []

# -----------------------------
# Genetischer Algorithmus
# -----------------------------
for generation in range(GENERATIONS):
    # Population speichern
    history.append(population.copy())

    # Bewertung & Sortierung
    population.sort(key=fitness, reverse=True)

    # Selektion: Top 50 %
    survivors = population[:POP_SIZE // 2]

    # Neue Population erzeugen
    new_population = survivors.copy()

    while len(new_population) < POP_SIZE:
        parent1, parent2 = random.sample(survivors, 2)

        # Crossover (arithmetisch)
        child = (parent1 + parent2) // 2

        # Mutation
        if random.random() < MUTATION_RATE:
            child += random.randint(-5, 5)

        # Wertebereich einhalten
        child = max(VALUE_RANGE[0], min(VALUE_RANGE[1], child))

        new_population.append(child)

    population = new_population

# -----------------------------
# Visualisierung
# -----------------------------
plt.figure(figsize=(10, 6))

for gen, pop in enumerate(history):
    plt.scatter(pop, [gen] * len(pop), s=12)

# Zielwert markieren
plt.axvline(TARGET, linestyle="--")

plt.xlabel("Lösungswert (Individuum)")
plt.ylabel("Generation")
plt.title("Genetischer Algorithmus: Annäherung der Population an das Optimum")
plt.grid(True)

# Optional: als Datei speichern (für Slides)
# plt.savefig("genetischer_algorithmus_population.png", dpi=200, bbox_inches="tight")

plt.show()