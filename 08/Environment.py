import numpy as np

class Environment:
    def __init__(self, goal_position, obstacle_positions, x):
        self.start_position = np.random.randint(x - 2, x - 1, size=2)
        self.goal_position = goal_position
        self.obstacle_positions = obstacle_positions
        self.grid = np.zeros((x, x))
        self.grid[goal_position] = 1
        for obstacle_position in obstacle_positions:
            self.grid[obstacle_position] = -1
        self.rows = x
        self.columns = x
        self.actions = ['up', 'down', 'left', 'right']
        self.state = None

    def reset(self):
        self.state = self.start_position
        return self.state

    def step(self, action):
        new_state = None

        if action == 'up':
            new_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 'down':
            new_state = (min(self.state[0] + 1, self.rows - 1), self.state[1])
        elif action == 'left':
            new_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 'right':
            new_state = (self.state[0], min(self.state[1] + 1, self.columns - 1))

        self.state = new_state

        if new_state == self.goal_position:
            reward = 1
            done = True
        elif new_state in self.obstacle_positions:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return new_state, reward, done
