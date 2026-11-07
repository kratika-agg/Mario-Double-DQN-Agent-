import random
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        experience = Transition(
            state, action, next_state, reward, done
        )
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)