# Mario Double DQN Agent 🍄

A Deep Reinforcement Learning project that trains an intelligent agent to play **Super Mario Bros** using **Double Deep Q-Network (Double DQN)**.

The agent learns to select actions from visual game observations using a CNN-based Q-network, experience replay, and a target network.

## Project Overview

This project implements a Double DQN agent for the `SuperMarioBros-1-1-v0` environment. It uses reinforcement learning to learn action-selection strategies through repeated interaction with the game environment.

### Key Features

- **Deep Q-Network (DQN):** Uses a Convolutional Neural Network (CNN) to predict Q-values for available actions.
- **Double DQN:** Uses the policy network to select the next action and the target network to evaluate it, helping reduce Q-value overestimation.
- **Experience Replay:** Stores past transitions and samples mini-batches for training.
- **Target Network:** Provides stable target Q-value estimates during learning.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation.
- **Frame Preprocessing:** Converts game observations into grayscale, resized, stacked frames.
- **Training Metrics:** Tracks episode rewards, average rewards, loss, epsilon, and global steps.
- **Gameplay Evaluation:** Loads a trained checkpoint and evaluates the agent using greedy action selection.
- **Training Visualization:** Generates reward and loss graphs to analyze learning behaviour.

## Double DQN Architecture

The agent follows the Double Deep Q-Network architecture, combining a policy network, a target network, and experience replay.

### Learning Workflow

1. **State Observation:** The agent receives the current game state as a stack of four preprocessed frames.
2. **Action Selection:** The policy network predicts Q-values for available actions. The agent selects an action using an epsilon-greedy strategy.
3. **Environment Interaction:** The selected action is performed in the Mario environment, producing the next state, reward, and termination information.
4. **Experience Storage:** The transition is stored in replay memory.
5. **Experience Sampling:** A mini-batch of transitions is sampled from replay memory.
6. **Double DQN Optimization:** The policy network selects the best next action, while the target network evaluates that action.
7. **Target Network Update:** The target network periodically copies the policy network's parameters.

### Core Components

| Component | Responsibility |
|---|---|
| Policy Network | Predicts Q-values and selects actions |
| Target Network | Calculates target Q-values |
| Replay Memory | Stores and samples past experiences |
| Mario Agent | Implements epsilon-greedy action selection |
| Environment | Provides game states, rewards, and transitions |

### Double DQN Optimization

Double DQN separates next-action selection from next-action evaluation:

- The **policy network** selects the action with the highest predicted Q-value.
- The **target network** evaluates the selected action.
- The resulting target is used to calculate the loss and update the policy network.