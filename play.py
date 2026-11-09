
import torch
import yaml

from environment import create_mario_env
from dqn import DQN
from agent import MarioAgent


# Load configuration
with open("parameters.yaml", "r") as file:
    config = yaml.safe_load(file)


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Initialize policy network
policy_net = DQN(
    input_channels=config["input_channels"],
    num_actions=config["num_actions"]
).to(device)


# Load trained model weights
policy_net.load_state_dict(
    torch.load(
        "mario_double_dqn.pth",
        map_location=device
    )
)

# Set network to evaluation mode
policy_net.eval()


# Initialize agent
agent = MarioAgent(
    policy_net,
    config["num_actions"],
    device=device
)


def convert_state(state):
    return torch.stack([
        frame.squeeze(0) if frame.ndim == 3 else frame
        for frame in state
    ]).float()


def play():

    env = create_mario_env(render_mode="human")

    try:
        state, info = env.reset()
        state = convert_state(state)

        total_reward = 0
        steps = 0
        max_steps = config["max_steps_per_episode"]

        for step in range(max_steps):

            # Select greedy action (no exploration)
            action = agent.select_action(state, epsilon=0)

            # Perform action
            next_state, reward, terminated, truncated, info = env.step(action)

            next_state = convert_state(next_state)

            total_reward += reward
            steps += 1

            state = next_state

            if terminated or truncated:
                break

        print("Game finished!")
        print("Total reward:", total_reward)
        print("Total steps:", steps)

    finally:
        env.close()


if __name__ == "__main__":
    play()
