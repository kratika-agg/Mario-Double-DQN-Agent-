
import random
import torch


class MarioAgent:

    def __init__(self, policy_net, num_actions, device="cpu"):
        self.policy_net = policy_net
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, epsilon):

        # Exploration: choose a random action
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        # Exploitation: choose the action with the highest Q-value
        state = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )

        # Convert (4, 1, 84, 84) into (1, 4, 84, 84)
        if state.ndim == 4:
            state = state.squeeze(1)

        state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state)
            #self.policy_net(state) passes the current state through our DQN.
            #q_values stores the predicted Q-values for each possible action.

        return q_values.argmax(dim=1).item()
