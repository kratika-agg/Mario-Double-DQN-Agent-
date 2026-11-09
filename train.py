
import torch
import torch.optim as optim
import yaml
import torch.nn as nn
from environment import create_mario_env
from agent import MarioAgent
import csv
import matplotlib.pyplot as plt

from dqn import DQN
from experience_replay import ReplayMemory


# Load configuration
with open("parameters.yaml", "r") as file:
    config = yaml.safe_load(file)


# Select computation device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Initialize online network
policy_net = DQN(
    input_channels=config["input_channels"],
    num_actions=config["num_actions"]
).to(device)


# Initialize target network
target_net = DQN(
    input_channels=config["input_channels"],
    num_actions=config["num_actions"]
).to(device)

# Copy online network weights to target network
target_net.load_state_dict(policy_net.state_dict())

# Set target network to evaluation mode
target_net.eval()


# Initialize optimizer
optimizer = optim.Adam(
    policy_net.parameters(),
    lr=config["learning_rate"]
)


# Initialize replay memory
memory = ReplayMemory(config["memory_size"])


# Load training parameters
BATCH_SIZE = config["batch_size"]
GAMMA = config["gamma"]
TARGET_UPDATE = config["target_update"]

print("Training components initialized successfully!")
print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", config["learning_rate"])

def convert_state(state):
    return torch.stack([
        frame.squeeze(0) if frame.ndim == 3 else frame
        for frame in state
    ]).float()

def optimize_model():

    # Check if enough experiences are available
    if len(memory) < BATCH_SIZE:
        return None
    #we need 32 smaples but if less exit fnc and return none without performing update    

    # Sample random experiences from replay memory
    transitions = memory.sample(BATCH_SIZE)

    # Convert experiences into a batch
    batch = list(zip(*transitions))

    # Extract states, actions, next states, rewards and done flags
    states = torch.stack(batch[0]).to(device)
    actions = torch.tensor(batch[1], dtype=torch.long, device=device)
    rewards = torch.tensor(batch[3], dtype=torch.float32, device=device)
    dones = torch.tensor(batch[4], dtype=torch.bool, device=device)

    # Predict Q-values for the current states
    current_q_values = policy_net(states)

    # Select Q-values corresponding to actions actually taken
    current_q_values = current_q_values.gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    # Initialize next-state values with zeros
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Identify experiences where the episode has not ended
    non_final_mask = ~dones

    # Get next states only for non-terminal experiences
    if non_final_mask.any():

        next_states = torch.stack([
            transition.next_state
            for transition, is_non_final
            in zip(transitions, non_final_mask)
            if is_non_final
        ]).to(device)    

        # Online network selects the best next action
        with torch.no_grad():

            next_actions = policy_net(next_states).argmax(
                dim=1, keepdim=True
            )

            # Target network evaluates the selected actions
            next_q_values = target_net(next_states).gather(
                1, next_actions
            ).squeeze(1)

            # Store the calculated future Q-values
            next_state_values[non_final_mask] = next_q_values

    # Calculate expected Q-values using Bellman equation
    expected_q_values = rewards + (
        GAMMA * next_state_values * (~dones).float()
    )


    # Calculate loss
    loss = nn.SmoothL1Loss()(
        current_q_values,
        expected_q_values
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Calculate gradients using backpropagation
    loss.backward()

    # Clip gradients to prevent excessively large updates
    torch.nn.utils.clip_grad_value_(
        policy_net.parameters(), 100
    )#Limits individual gradients to the range -100 to 100.

    # Update online network weights
    optimizer.step()

    # Return loss value
    return loss.item()    


def train():

    # Create the Mario environment
    env = create_mario_env()

    # Initialize the agent
    agent = MarioAgent(
        policy_net,
        config["num_actions"],
        device=device
    )

    # Initialize epsilon for exploration
    epsilon = config["epsilon_start"]

    # Load training parameters
    epsilon_end = config["epsilon_end"]
    epsilon_decay = config["epsilon_decay"]
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps_per_episode"]

    # Track total environment steps
    global_steps = 0

    # Store rewards from all episodes
    episode_rewards = []


    with open("training_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Episode",
            "Reward",
            "Average Reward (100)",
            "Average Loss",
            "Epsilon",
            "Global Steps"
        ])

    try:

        # Run training episodes
        for episode in range(num_episodes):

            # Reset environment at the beginning of each episode
            state, info = env.reset()

            state = convert_state(state)

            # Initialize episode reward
            total_reward = 0
            episode_losses = []

            # Run one episode
            for step in range(max_steps):

                # Select action using epsilon-greedy strategy
                action = agent.select_action(state, epsilon)

                # Perform action in the environment
                next_state, reward, terminated, truncated, info = env.step(action)

                # Convert LazyFrames observations into PyTorch tensors
                next_state = convert_state(next_state)

                # Determine whether the environment has ended the episode
                done = terminated or truncated

                # Check whether our own step limit has been reached
                time_limit_reached = step == max_steps - 1

                # Determine whether to stop the episode
                episode_done = done or time_limit_reached

                # Store the experience
                memory.push(
                    state,
                    action,
                    None if terminated else next_state,
                    reward,
                    terminated
                )

                # Update current state
                state = next_state

                # Accumulate episode reward
                total_reward += reward

                # Optimize policy network
                loss = optimize_model()
                if loss is not None:
                    episode_losses.append(loss)

                # Count environment steps
                global_steps += 1

                # Periodically synchronize target network
                if global_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(
                        policy_net.state_dict()
                    )
                    target_net.eval()
                # Stop episode if it has ended
                if episode_done:
                    break

            # Store episode reward
            episode_rewards.append(total_reward)
            
            avg_reward = sum(
                episode_rewards[-100:]
            ) / len(episode_rewards[-100:])
            # Decay epsilon after each episode
            epsilon = max(
                epsilon_end,
                epsilon * epsilon_decay
            )

            avg_loss = (
                sum(episode_losses) / len(episode_losses)
                if episode_losses else 0
            )

            

            with open("training_metrics.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    episode + 1,
                    total_reward,
                    avg_reward,
                    avg_loss,
                    epsilon,
                    global_steps
                ])

            # Display episode progress
            print(
                f"Episode: {episode + 1}/{num_episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Steps: {global_steps} | "
                f"Average Loss: {avg_loss:.4f} | "
            )

    finally:
        # Close the environment
        env.close()

    # Save trained policy network
    torch.save(
        policy_net.state_dict(),
        "mario_double_dqn.pth"
    )

    print("Training completed!")
    print("Model saved successfully!")


if __name__ == "__main__":
    train()

