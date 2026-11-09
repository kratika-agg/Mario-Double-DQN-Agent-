# Mario Double DQN Agent

An implementation of a **Double Deep Q-Network (Double DQN)** agent designed to learn to play Super Mario Bros using Deep Reinforcement Learning and Convolutional Neural Networks.

## Project Overview

This project focuses on developing an intelligent agent that learns to interact with the Super Mario Bros environment through reinforcement learning.

The agent uses visual observations of the game environment to estimate action values and make decisions.

The implementation is based on Double DQN, which uses separate online and target networks to improve the stability of Q-learning.

## Tech Stack

- Python
- PyTorch
- NumPy
- OpenAI Gym
- Gym Super Mario Bros
- TorchVision
- TensorDict
- TorchRL
- PyYAML

## Project Architecture

The project is organized into modular components:

| File | Responsibility |
|---|---|
| `environment.py` | Creates and preprocesses the Mario environment |
| `dqn.py` | Defines the CNN-based Q-network |
| `experience_replay.py` | Implements experience replay memory |
| `agent.py` | Handles action selection using epsilon-greedy policy |
| `parameters.yaml` | Stores configurable hyperparameters |
| `train.py` | Initializes training components and implements the optimization step |

## Implemented Components

### 1. Environment Setup and Preprocessing

- Initialized the Super Mario Bros environment.
- Restricted the action space to two actions: moving right and jumping right.
- Implemented frame skipping to reduce computational overhead.
- Converted observations into grayscale.
- Resized frames to 84 × 84.
- Applied frame stacking to provide temporal information to the agent.

### 2. Deep Q-Network

Implemented a Convolutional Neural Network using PyTorch.

The network processes visual observations and predicts Q-values for the available actions.

### 3. Experience Replay

Implemented replay memory using Python's `deque`.

It stores transitions containing:

- Current state
- Action
- Next state
- Reward
- Terminal flag

Random sampling is used to obtain batches of experiences for training.

### 4. Agent and Action Selection

Implemented an epsilon-greedy action selection strategy.

- Exploration: Selects a random action.
- Exploitation: Selects the action with the highest predicted Q-value.

### 5. Configuration Management

Training parameters are maintained separately in `parameters.yaml`, making it easier to modify hyperparameters without changing the implementation.

## Current Development Status

- [x] Environment setup and preprocessing
- [x] CNN-based Q-network
- [x] Experience replay memory
- [x] Epsilon-greedy action selection
- [x] Configuration management
- [x] Online and target network initialization
- [x] Double DQN optimization step
- [ ] Complete training loop
- [ ] Periodic target network updates
- [ ] Model checkpointing
- [ ] Training evaluation and performance analysis

## Project Goal

Build and evaluate a Double DQN agent capable of learning effective gameplay strategies through interaction with the Super Mario Bros environment.

---

**Status:** Under Development