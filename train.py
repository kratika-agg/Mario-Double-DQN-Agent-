
import torch
import torch.optim as optim
import yaml

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
