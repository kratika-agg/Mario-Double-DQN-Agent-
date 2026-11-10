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

def enable_fullscreen(env):

    current_env = env

    print("\n--- Environment Wrapper Chain ---")

    while True:
        print(type(current_env))

        if hasattr(current_env, "viewer"):
            print("Viewer found!")
            print("Viewer:", current_env.viewer)

            if current_env.viewer is not None:
                current_env.viewer._window.set_fullscreen(True)
                print("Fullscreen enabled!")
                return

        if not hasattr(current_env, "env"):
            break

        current_env = current_env.env

    print("Viewer was not found in the wrapper chain.")
    base_env = env.unwrapped

    # Access the original NES environment
    nes_env = base_env.env

    # Access the Pyglet viewer
    viewer = nes_env.viewer

    # Enable fullscreen
    viewer._window.set_fullscreen(True)
    base_env = env.unwrapped

    # Access the NES image viewer
    viewer = base_env.viewer

    # Enable fullscreen on the existing Pyglet window
    viewer._window.set_fullscreen(True)

def play():

    env = create_mario_env(render_mode="human")

    try:
        state, info = env.reset()
        state = convert_state(state)
        # Create the rendering window
        env.render()

        # Enable fullscreen
        enable_fullscreen(env)

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
