
"""
play.py

Loads the trained Double DQN model and evaluates its
gameplay performance over multiple episodes.
Supports fullscreen rendering through the Pyglet viewer.
"""

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
    torch.load("mario_double_dqn_100ep.pth", map_location=device)
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


def enable_fullscreen(env):

    current_env = env

    while True:

        if hasattr(current_env, "viewer"):

            if current_env.viewer is not None:
                current_env.viewer._window.set_fullscreen(True)
                print("Fullscreen enabled!")
                return

        if not hasattr(current_env, "env"):
            break

        current_env = current_env.env

    raise RuntimeError("Could not find the Pyglet viewer.")


def play():

    env = create_mario_env(render_mode="human")

    try:

        num_episodes = 5
        max_steps = config["max_steps_per_episode"]

        for episode in range(1, num_episodes + 1):

            state, info = env.reset()
            state = convert_state(state)

            # Initialize fullscreen only once
            if episode == 1:
                env.render()
                enable_fullscreen(env)

            total_reward = 0
            steps = 0

            for step in range(max_steps):

                # Greedy action selection
                with torch.no_grad():
                    action = agent.select_action(state, epsilon=0)

                # Perform action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                next_state = convert_state(next_state)

                total_reward += reward
                steps += 1

                state = next_state

                if terminated or truncated:
                    break

            # Identify episode termination reason
            if terminated:
                reason = "Terminated"

            elif truncated:
                reason = "Truncated"

            else:
                reason = "Maximum steps reached"

            print(
                f"Episode: {episode}/{num_episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"X Position: {info.get('x_pos', 0)} | "
                f"Flag: {info.get('flag_get', False)} | "
                f"Reason: {reason}"
            )

        print("\nEvaluation completed!")

    finally:
        env.close()


if __name__ == "__main__":
    play()
