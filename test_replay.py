test_replay.py — verifies replay memory.
from experience_replay import ReplayMemory

memory = ReplayMemory(capacity=5)

for i in range(7):
    memory.push(
        state=f"state_{i}",
        action=i,
        next_state=f"next_state_{i}",
        reward=float(i),
        done=False
    )

print("Memory size:", len(memory))

batch = memory.sample(3)

print("\nSampled experiences:")

for experience in batch:
    print(experience)