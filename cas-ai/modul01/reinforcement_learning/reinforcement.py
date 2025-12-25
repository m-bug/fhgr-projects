import numpy as np
import matplotlib.pyplot as plt

###
# Getting started:
#
# pip install numpy matplotlib
#
###

# ------------------------
# 1. Setup
# ------------------------
np.random.seed(42)

n_states = 3     # Kundentypen (z.B. aus Unsupervised Learning:
                 # 0 = digitale Vielnutzer
                 # 1 = klassische Kunden
                 # 2 = vermögende Kunden)

n_actions = 3    # Beispiel-Angebote:
                 # 0 = Angebot A
                 # 1 = Angebot B
                 # 2 = kein Angebot

episodes = 1000   # Anzahl Interaktionen

# Q-Tabelle: Zustand x Aktion
Q = np.zeros((n_states, n_actions))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2  # Explorationsrate (ε-greedy)

# ------------------------
# 2. Reward-Funktion (simuliertes Kundenverhalten)
# ------------------------
def get_reward(state, action):
    """
    Simuliert Kundenreaktionen auf Angebote.
    Rewards:
    +1  = positives Feedback
     0  = neutral
    -1  = negativ
    """
    preferences = {
        0: [ 1,  0, -1],   # digitaler Vielnutzer
        1: [ 0,  1, -1],   # klassischer Kunde
        2: [-1,  1,  0]    # vermögender Kunde
    }
    return preferences[state][action]

# ------------------------
# 3. Training (Q-Learning)
# ------------------------
reward_history = []

for _ in range(episodes):
    # zufälliger Kunde / Zustand
    state = np.random.randint(0, n_states)

    # ε-greedy Strategie
    if np.random.rand() < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q[state])

    reward = get_reward(state, action)
    reward_history.append(reward)

    # Q-Learning Update
    Q[state, action] += learning_rate * (
        reward
        + discount_factor * np.max(Q[state])
        - Q[state, action]
    )

# ------------------------
# 4. Visualisierung
# ------------------------
cumulative_reward = np.cumsum(reward_history)

# Lineare Referenz:
# Annahme: konstanter durchschnittlicher Reward pro Interaktion
avg_reward_reference = 1.0
linear_reference = avg_reward_reference * np.arange(len(cumulative_reward))

plt.figure(figsize=(7, 5))

# RL-Lernkurve
plt.plot(
    cumulative_reward,
    label="Reinforcement Learning (kumulativer Reward)",
    linewidth=2
)

# Lineare Vergleichslinie
plt.plot(
    linear_reference,
    linestyle="--",
    color="red",
    label="Theoretische lineare Steigung (konstanter Reward)"
)

plt.xlabel("Interaktionen")
plt.ylabel("Kumulativer Reward")
plt.title("Reinforcement Learning: Lernverlauf vs. lineare Referenz")
plt.grid(True)
plt.legend()

# Annotation zur Erklärung
plt.text(
    0.05 * len(cumulative_reward),
    0.75 * max(cumulative_reward),
    "RL-Lernprozess:\n"
    "• anfänglich Exploration\n"
    "• schwankende Rewards\n"
    "• langfristig annähernd lineares Wachstum",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85)
)

plt.tight_layout()

# ------------------------
# 5. Export für Beamer
# ------------------------
plt.savefig("reinforcement_learning_vs_linear.png", dpi=300)
plt.show()

