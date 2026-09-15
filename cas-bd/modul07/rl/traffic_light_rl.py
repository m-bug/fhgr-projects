"""
Ampel-Bot: Ein Reinforcement-Learning-Agent lernt eine Kreuzung zu steuern.

- Kreuzung: eine einfache Kreuzung mit 4 Zufahrten (Nord, Sued, Ost, West)
- Agent: tabellarisches Q-Learning
- Visualisierung: Pygame (Live-Ansicht waehrend des Trainings)

Steuerung waehrend der Ausfuehrung:
  LEERTASTE  -> zwischen "schnell" (kein Rendering) und "live" umschalten
  ESC / Fenster schliessen -> Programm beenden

Getting started:
    python -m venv venv
    venv\Scripts\activate      # Windows
    source venv/bin/activate   # macOS/Linux
    pip install pygame
    pip install pygame-ce

Run:  python traffic_light_rl.py
"""

import random
import sys
from collections import defaultdict

import pygame

# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------
EPISODES = 500
STEPS_PER_EPISODE = 200
ARRIVAL_PROB = 0.30        # Wahrscheinlichkeit, dass pro Schritt ein Auto ankommt
DEPARTURE_CAPACITY = 2     # Autos, die pro Schritt bei Gruen abfliessen koennen
SWITCH_PENALTY = 8         # zusaetzliche Strafe fuer einen Ampelwechsel (Gelbphase = Stillstand)
MAX_QUEUE_FOR_STATE = 8    # Deckelung der Warteschlange fuer die Zustandsdiskretisierung
BIN_SIZE = 2               # Groesse der Bins fuer die Diskretisierung

# Q-Learning Hyperparameter
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.985

# Rendering
RENDER_EVERY = 10          # nur jede N-te Episode live zeigen (Rest laeuft im Schnelldurchlauf)
RENDER_LAST_N = 20         # die letzten N Episoden immer zeigen (gelernte Strategie)
FPS_LIVE = 30
WINDOW_SIZE = 700

DIRECTIONS = ["N", "S", "E", "W"]


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
class TrafficEnv:
    """Sehr einfaches Kreuzungsmodell mit zwei Ampelphasen.

    Phase 0: Nord-Sued hat Gruen (Ost-West steht)
    Phase 1: Ost-West hat Gruen (Nord-Sued steht)
    """

    def __init__(self):
        self.queues = {d: 0 for d in DIRECTIONS}
        self.phase = 0

    def reset(self):
        self.queues = {d: 0 for d in DIRECTIONS}
        self.phase = 0
        return dict(self.queues), self.phase

    def step(self, action):
        switched = action != self.phase
        self.phase = action

        # Ankunft neuer Autos an jeder Zufahrt
        for d in DIRECTIONS:
            if random.random() < ARRIVAL_PROB:
                self.queues[d] += 1

        # Abfluss nur, wenn nicht gerade gewechselt wurde (Gelbphase = kein Durchfluss)
        if not switched:
            green_dirs = ("N", "S") if self.phase == 0 else ("E", "W")
            for d in green_dirs:
                depart = min(self.queues[d], DEPARTURE_CAPACITY)
                self.queues[d] -= depart

        reward = -sum(self.queues.values())
        if switched:
            reward -= SWITCH_PENALTY

        return dict(self.queues), self.phase, reward, switched


def discretize(queues, phase):
    """Reduziert die (potenziell unbegrenzten) Warteschlangen auf ein handhabbares Zustandsraster."""
    binned = tuple(min(queues[d], MAX_QUEUE_FOR_STATE) // BIN_SIZE for d in DIRECTIONS)
    return binned + (phase,)


# ----------------------------------------------------------------------
# Q-Learning Agent
# ----------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, n_actions=2):
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: [0.0] * n_actions)
        self.epsilon = EPSILON_START

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q_values = self.q_table[state]
        return max(range(self.n_actions), key=lambda a: q_values[a])

    def update(self, state, action, reward, next_state):
        best_next = max(self.q_table[next_state])
        td_target = reward + GAMMA * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += ALPHA * td_error

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ----------------------------------------------------------------------
# Visualisierung
# ----------------------------------------------------------------------
COLOR_BG = (30, 30, 35)
COLOR_ROAD = (60, 60, 65)
COLOR_RED = (200, 60, 60)
COLOR_GREEN = (60, 200, 100)
COLOR_CAR = (240, 200, 60)
COLOR_TEXT = (230, 230, 230)

CENTER = (WINDOW_SIZE // 2, WINDOW_SIZE // 2)
ROAD_WIDTH = 90
MAX_CARS_DRAWN = 14
CAR_SIZE = 16
CAR_GAP = 4


def draw_scene(screen, font, env, episode, step, total_reward, epsilon, mode_text):
    screen.fill(COLOR_BG)
    cx, cy = CENTER

    # Strassen (Kreuz)
    pygame.draw.rect(screen, COLOR_ROAD, (0, cy - ROAD_WIDTH // 2, WINDOW_SIZE, ROAD_WIDTH))
    pygame.draw.rect(screen, COLOR_ROAD, (cx - ROAD_WIDTH // 2, 0, ROAD_WIDTH, WINDOW_SIZE))

    ns_color = COLOR_GREEN if env.phase == 0 else COLOR_RED
    ew_color = COLOR_GREEN if env.phase == 1 else COLOR_RED

    # Ampeln (kleine Kreise an den vier Zufahrten)
    pygame.draw.circle(screen, ns_color, (cx - ROAD_WIDTH // 2 - 20, cy - ROAD_WIDTH // 2 - 20), 10)
    pygame.draw.circle(screen, ns_color, (cx + ROAD_WIDTH // 2 + 20, cy + ROAD_WIDTH // 2 + 20), 10)
    pygame.draw.circle(screen, ew_color, (cx + ROAD_WIDTH // 2 + 20, cy - ROAD_WIDTH // 2 - 20), 10)
    pygame.draw.circle(screen, ew_color, (cx - ROAD_WIDTH // 2 - 20, cy + ROAD_WIDTH // 2 + 20), 10)

    # Warteschlangen als gestapelte Rechtecke ("Autos")
    def draw_queue(direction, count):
        count = min(count, MAX_CARS_DRAWN)
        for i in range(count):
            offset = ROAD_WIDTH // 2 + 15 + i * (CAR_SIZE + CAR_GAP)
            if direction == "N":
                rect = (cx - CAR_SIZE // 2, cy - offset - CAR_SIZE, CAR_SIZE, CAR_SIZE)
            elif direction == "S":
                rect = (cx - CAR_SIZE // 2, cy + offset, CAR_SIZE, CAR_SIZE)
            elif direction == "W":
                rect = (cx - offset - CAR_SIZE, cy - CAR_SIZE // 2, CAR_SIZE, CAR_SIZE)
            else:  # "E"
                rect = (cx + offset, cy - CAR_SIZE // 2, CAR_SIZE, CAR_SIZE)
            pygame.draw.rect(screen, COLOR_CAR, rect, border_radius=3)

    for d in DIRECTIONS:
        draw_queue(d, env.queues[d])

    # HUD
    lines = [
        f"Episode: {episode + 1}/{EPISODES}   Schritt: {step + 1}/{STEPS_PER_EPISODE}",
        f"Belohnung (Episode): {total_reward:.0f}   Epsilon: {epsilon:.3f}",
        f"Warteschlangen  N:{env.queues['N']}  S:{env.queues['S']}  "
        f"E:{env.queues['E']}  W:{env.queues['W']}",
        mode_text,
    ]
    for i, line in enumerate(lines):
        text_surf = font.render(line, True, COLOR_TEXT)
        screen.blit(text_surf, (10, 10 + i * 20))

    pygame.display.flip()


# ----------------------------------------------------------------------
# Training + Live-Loop
# ----------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Ampel-Bot - Reinforcement Learning")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    env = TrafficEnv()
    agent = QLearningAgent()

    live_rendering = True
    episode_rewards = []

    for episode in range(EPISODES):
        render_this_episode = (
            episode % RENDER_EVERY == 0 or episode >= EPISODES - RENDER_LAST_N
        )

        queues, phase = env.reset()
        state = discretize(queues, phase)
        total_reward = 0

        for step in range(STEPS_PER_EPISODE):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_SPACE:
                        live_rendering = not live_rendering

            action = agent.get_action(state)
            queues, phase, reward, switched = env.step(action)
            next_state = discretize(queues, phase)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if render_this_episode and live_rendering:
                mode = "Live-Ansicht (LEERTASTE = Schnelldurchlauf)"
                draw_scene(screen, font, env, episode, step, total_reward, agent.epsilon, mode)
                clock.tick(FPS_LIVE)
            elif render_this_episode:
                # Schnelldurchlauf, aber Fenster bleibt reaktionsfaehig
                if step % 20 == 0:
                    mode = "Schnelldurchlauf (LEERTASTE = Live)"
                    draw_scene(screen, font, env, episode, step, total_reward, agent.epsilon, mode)

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg = sum(episode_rewards[-50:]) / 50
            print(f"Episode {episode + 1:4d}  |  Ø Belohnung (letzte 50): {avg:8.1f}  |  Epsilon: {agent.epsilon:.3f}")

    print("Training abgeschlossen. Fenster zeigt weiter die gelernte Strategie (ESC zum Beenden).")

    # Nach dem Training: gelernte Strategie in Dauerschleife weiter zeigen, mit ESC beenden
    while True:
        queues, phase = env.reset()
        state = discretize(queues, phase)
        for step in range(STEPS_PER_EPISODE):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

            action = agent.get_action(state)  # noch mit kleinem Epsilon, meist aber "greedy"
            queues, phase, reward, switched = env.step(action)
            state = discretize(queues, phase)
            draw_scene(screen, font, env, EPISODES - 1, step, 0, agent.epsilon, "Gelernte Strategie (ESC zum Beenden)")
            clock.tick(FPS_LIVE)


if __name__ == "__main__":
    main()