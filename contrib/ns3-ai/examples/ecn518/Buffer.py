import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state= zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        return len(self.buffer)