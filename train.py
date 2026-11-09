
import torch
import torch.optim as optim
import yaml
import torch.nn as nn
from environment import create_mario_env
from agent import MarioAgent


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
    expected_q_values = rewards + (GAMMA * next_state_values)

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