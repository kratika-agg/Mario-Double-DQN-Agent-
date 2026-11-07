
import torch

from dqn import DQN
from agent import MarioAgent


def test_agent():

    # Initialize the online DQN
    policy_net = DQN()

    # Create the Mario agent
    agent = MarioAgent(
        policy_net=policy_net,
        num_actions=2
    )

    # Generate a dummy Mario state
    state = torch.rand(4, 1, 84, 84)

    # Test 1: Exploration
    exploration_action = agent.select_action(
        state,
        epsilon=1.0
    )

    print("Exploration action:", exploration_action)
    assert exploration_action in [0, 1]
    print("Exploration test passed!")

    # Test 2: Exploitation
    exploitation_action = agent.select_action(
        state,
        epsilon=0.0
    )

    print("Exploitation action:", exploitation_action)
    assert exploitation_action in [0, 1]
    print("Exploitation test passed!")


if __name__ == "__main__":
    test_agent()
