import numpy as np
import pygame
from Environment import Environment


class QLearningAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9, initial_exploration_prob=1.0,
                 min_exploration_prob=0.1, decay_rate=0.995):
        self.environment = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_prob = initial_exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.decay_rate = decay_rate
        self.Q_table = np.random.rand(environment.rows, environment.columns, len(environment.actions))
        self.episode_number = 0

    def train(self, num_episodes):
        pygame.init()

        screen_size = (self.environment.rows * 100, self.environment.columns * 100 + 40)
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Grid World Q-Values Visualization")

        running = True
        font = pygame.font.Font(None, 36)
        progress_font = pygame.font.Font(None, 24)

        frame_delay = 5

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if self.episode_number < num_episodes:
                state = self.environment.reset()
                done = False

                while not done:

                    current_explore_prob = max(self.min_exploration_prob,
                                               self.initial_exploration_prob * (self.decay_rate ** self.episode_number))

                    if np.random.rand() < current_explore_prob:
                        action = np.random.choice(self.environment.actions)
                    else:
                        action = self.select_best_action(state)

                    new_state, reward, done = self.environment.step(action)
                    self.update_q_table(state, action, reward, new_state)
                    state = new_state

                self.episode_number += 1

            # Display Q-values in real-time
            for row in range(self.environment.rows):
                for col in range(self.environment.columns):
                    state = (row, col)
                    q_values = self.Q_table[row, col]
                    max_q_value = np.max(q_values)
                    action_index = np.argmax(q_values)

                    # Draw the grid cell
                    pygame.draw.rect(screen, (255, 255, 255), (col * 100, row * 100, 100, 100))
                    pygame.draw.rect(screen, (0, 0, 0), (col * 100, row * 100, 100, 100), 1)

                    # Draw an arrow representing the best action based on Q-value
                    if max_q_value > 0:
                        if action_index == 0:  # Up
                            pygame.draw.rect(screen, (175, 255, 255 - min(255, max_q_value * 255)),
                                             (col * 100, row * 100, 100, 100))
                            pygame.draw.polygon(screen, (255, 0, 0), [(col * 100 + 50, row * 100 + 20),
                                                                      (col * 100 + 40, row * 100 + 40),
                                                                      (col * 100 + 60, row * 100 + 40)])
                        elif action_index == 1:  # Down
                            pygame.draw.rect(screen, (175, 255, 255 - min(255, max_q_value * 255)),
                                             (col * 100, row * 100, 100, 100))
                            pygame.draw.polygon(screen, (255, 0, 0), [(col * 100 + 50, row * 100 + 80),
                                                                      (col * 100 + 40, row * 100 + 60),
                                                                      (col * 100 + 60, row * 100 + 60)])
                        elif action_index == 2:  # Left
                            pygame.draw.rect(screen, (175, 255, 255 - min(255, max_q_value * 255)),
                                             (col * 100, row * 100, 100, 100))
                            pygame.draw.polygon(screen, (255, 0, 0), [(col * 100 + 20, row * 100 + 50),
                                                                      (col * 100 + 40, row * 100 + 40),
                                                                      (col * 100 + 40, row * 100 + 60)])
                        elif action_index == 3:  # Right
                            pygame.draw.rect(screen, (175, 255, 255 - min(255, max_q_value * 255)),
                                             (col * 100, row * 100, 100, 100))
                            pygame.draw.polygon(screen, (255, 0, 0), [(col * 100 + 80, row * 100 + 50),
                                                                      (col * 100 + 60, row * 100 + 40),
                                                                      (col * 100 + 60, row * 100 + 60)])

                    # Display the max Q-value on the cell
                    text = font.render(f"{max_q_value:.2f}", True, (0, 0, 0))
                    screen.blit(text, (col * 100 + 10, row * 100 + 10))

                    # Draw goal, obstacle, and start positions
                    if np.array_equal(state, self.environment.goal_position):
                        # Draw a cross for the goal
                        pygame.draw.line(screen, (255, 0, 0), (col * 100 + 20, row * 100 + 20),
                                         (col * 100 + 80, row * 100 + 80), 5)
                        pygame.draw.line(screen, (255, 0, 0), (col * 100 + 20, row * 100 + 80),
                                         (col * 100 + 80, row * 100 + 20), 5)
                    elif state in self.environment.obstacle_positions:
                        # Draw a solid square for the obstacle
                        pygame.draw.rect(screen, (0, 0, 0), (col * 100, row * 100, 100, 100))
                    elif np.array_equal(state, self.environment.start_position):
                        # Draw a circle for the start
                        pygame.draw.circle(screen, (0, 255, 0), (col * 100 + 50, row * 100 + 50), 30)

            # Clear the area for progress text
            pygame.draw.rect(screen, (255, 255, 255), (0, self.environment.rows * 100, screen_size[0], 40))

            # Display progress text
            progress_text = progress_font.render(f"Episodes: {self.episode_number}/{num_episodes}", True, (0, 0, 0))
            screen.blit(progress_text, (10, self.environment.rows * 100 + 10))
            progress_bar_length = (self.episode_number / num_episodes) * (screen_size[0] - 20)
            pygame.draw.rect(screen, (0, 0, 255), (10, self.environment.rows * 100 + 30, progress_bar_length, 10))

            pygame.display.flip()  # Update the display in real-time

            pygame.time.delay(frame_delay)

        pygame.quit()

    def update_q_table(self, state, action, reward, new_state):
        current_q_value = self.Q_table[state[0], state[1], self.environment.actions.index(action)]
        max_future_q_value = np.max(self.Q_table[new_state[0], new_state[1]])

        new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (
                reward + self.discount_factor * max_future_q_value)

        self.Q_table[state[0], state[1], self.environment.actions.index(action)] = new_q_value

    def select_best_action(self, state):
        return self.environment.actions[np.argmax(self.Q_table[state[0], state[1]])]


def test_run(num_episodes):
    goal_position = (1, 6)
    obstacle_positions = [(2, 2), (6, 2), (4, 4), (3, 2), (4, 0),
                          (2, 3), (2, 1), (2, 3), (2, 6),
                          (6, 3), (1, 4), (4, 5), (1, 5),
                          (5, 4), (6, 4)]
    env = Environment(goal_position, obstacle_positions, 7)

    agent = QLearningAgent(env)
    agent.train(num_episodes=num_episodes)


if __name__ == "__main__":
    x = 7
    num_obstacles = 20
    num_episodes = 2000

    test_run(num_episodes)
