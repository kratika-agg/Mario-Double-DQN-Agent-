
from environment import create_mario_env

env = create_mario_env()

state, info = env.reset()

print("Environment initialized successfully!")
print("State shape:", state.shape)
print("State dtype:", state.dtype)

# Test one environment step
action = 0

next_state, reward, terminated, truncated, info = env.step(action)

print("\nStep test successful!")
print("Next state shape:", next_state.shape)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()
