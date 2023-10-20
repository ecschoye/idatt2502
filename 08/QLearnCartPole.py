import gym
import numpy as np

class QLearnCartPole:
    def __init__(self, episodes, learning_rate, gamma, epsilon, epsilon_decay):
        self.env = gym.make("CartPole-v1")
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.bin_size = 30
        self.state_bins = self.create_bins()
        self.qtable = np.zeros([self.bin_size] * len(self.state_bins) + [self.env.action_space.n])
        self.best_qtable = None
        self.best_score = 0

    def create_bins(self):
        state_bins = [
            np.linspace(-4.8, 4.8, self.bin_size),
            np.linspace(-5, 5, self.bin_size),
            np.linspace(-0.418, 0.418, self.bin_size),
            np.linspace(-5, 5, self.bin_size)
        ]
        return state_bins

    def discretize_state(self, state):
        return tuple(np.digitize(state[i], self.state_bins[i]) - 1 for i in range(len(state)))

    def select_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[state])

    def q_learning_update(self, state, action, reward, observation):
        return (1 - self.learning_rate) * self.qtable[state][action] + self.learning_rate * (
                reward + self.gamma * np.max(self.qtable[observation]))

    def train_cartpole(self):
        rewards = 0
        steps = 0

        for episode in range(1, self.episodes):
            state, info = self.env.reset()
            state = self.discretize_state(state)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                observation, reward, done, _, _ = self.env.step(action)
                observation = self.discretize_state(observation)
                score += reward

                self.qtable[state][action] = self.q_learning_update(state, action, reward, observation)
                state = observation

            rewards += score
            steps += 1

            if episode % 100 == 0:
                print(f"Episode {episode}, Final Score: {score}, Epsilon: {self.epsilon}")

            if score > 475 and steps > 100:
                if score > self.best_score:
                    self.best_score = score
                    self.best_qtable = np.copy(self.qtable)


        self.env.close()

    def test_cartpole(self, num_episodes):
        self.env = gym.make("CartPole-v1", render_mode="human")
        if self.best_qtable is None:
            print("The model has not been trained yet!")
            return

        for episode in range(num_episodes):
            state, info = self.env.reset()
            state = self.discretize_state(state)
            done = False
            score = 0

            while not done:
                action = np.argmax(self.best_qtable[state])
                observation, reward, done, _, _ = self.env.step(action)
                observation = self.discretize_state(observation)
                score += reward
                state = observation

            print(f"Test Episode {episode + 1}, Score: {score}")

if __name__ == "__main__":
    episodes = 10000
    test_episodes = 10
    learning_rate = 0.15
    gamma = 0.995
    epsilon = 0.2
    epsilon_decay = 0.999

    qlearning = QLearnCartPole(episodes, learning_rate, gamma, epsilon, epsilon_decay)
    qlearning.train_cartpole()
    qlearning.test_cartpole(test_episodes)
