import gym
import numpy as np


def create_bins(bin_size):
    # Create bins to discretize the continuous state space
    state_bins = [
        np.linspace(-4.8, 4.8, bin_size),
        np.linspace(-5, 5, bin_size),
        np.linspace(-0.418, 0.418, bin_size),
        np.linspace(-5, 5, bin_size)
    ]
    return state_bins


def discretize_state(state, state_bins):
    # Discretize the continuous state space using the bins created
    return tuple(np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state)))


def select_action(qtable, state, epsilon, env):
    # Select an action using epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(qtable[state])  # Exploit learned values


def q_learning_update(qtable, state, action, reward, observation, learning_rate, gamma):
    # Update Q-table based on Q-learning equation
    return qtable[state][action] + learning_rate * (
            reward + gamma * np.max(qtable[observation]) - qtable[state][action])


def train_cartpole():
    env = gym.make("CartPole-v1", render_mode="human")
    state_space = 4
    action_space = 2
    bin_size = 30

    qtable = np.random.uniform(low=-1, high=1, size=([bin_size] * state_space + [action_space]))
    state_bins = create_bins(bin_size)

    episodes = 10000
    learning_rate = 0.15
    gamma = 0.995
    epsilon = 0.1

    rewards = 0
    steps = 0

    for episode in range(1, episodes):
        steps += 1
        state, info = env.reset()
        state = discretize_state(state, state_bins)
        score = 0
        terminated = False

        while not terminated:
            if episode % 500 == 0:
                print(f"Episode {episode}, Score: {score}, Epsilon: {epsilon}")
                env.render()

            action = select_action(qtable, state, epsilon, env)

            observation, reward, terminated, _, _ = env.step(action)
            observation = discretize_state(observation, state_bins)
            score += reward

            if not terminated:
                # Update Q-table based on Q-learning equation
                qtable[state][action] = q_learning_update(qtable, state, action, reward, observation, learning_rate,
                                                          gamma)
            state = observation

        rewards += score
        if score > 195 and steps > 100:
            print(f"Solved in {episode} episodes")
            break

    env.close()


if __name__ == "__main__":
    train_cartpole()
