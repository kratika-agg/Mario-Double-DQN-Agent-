
import yaml

# Open the YAML configuration file
with open("parameters.yaml", "r") as file:
    config = yaml.safe_load(file)

# Display the configuration
print("Configuration loaded successfully!")

print("Environment:", config["env_id"])
print("Learning rate:", config["learning_rate"])
print("Batch size:", config["batch_size"])
print("Number of episodes:", config["num_episodes"])

