
import csv
import matplotlib.pyplot as plt


# Lists to store training metrics
episodes = []
rewards = []
average_rewards = []
losses = []


# Read CSV file
with open("training_metrics.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        episodes.append(int(row["Episode"]))
        rewards.append(float(row["Reward"]))
        average_rewards.append(
            float(row["Average Reward (100)"])
        )
        losses.append(float(row["Average Loss"]))


# Plot rewards
plt.figure(figsize=(12, 5))

plt.plot(
    episodes,
    rewards,
    alpha=0.4,
    label="Episode Reward"
)

plt.plot(
    episodes,
    average_rewards,
    label="Average Reward (100 Episodes)"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Mario Double DQN - Training Rewards")
plt.legend()
plt.grid(True)

plt.savefig("reward_curve.png")
plt.show()


# Plot training loss
plt.figure(figsize=(12, 5))

plt.plot(
    episodes,
    losses,
    label="Average Training Loss"
)

plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Mario Double DQN - Training Loss")
plt.legend()
plt.grid(True)

plt.savefig("training_loss.png")
plt.show()
