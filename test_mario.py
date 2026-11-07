from environment import create_mario_env

env = create_mario_env()

state, info = env.reset()

print("Environment initialized successfully!")
print("State shape:", state.shape)
print("State dtype:", state.dtype)

env.close()