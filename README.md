Mario Double DQN Agent

A reinforcement learning project that explores training an agent to play Super Mario Bros. using Double Deep Q-Networks (Double DQN) and PyTorch.

Overview

The agent learns to interact with the Mario environment by observing game states, selecting actions, and learning from the rewards received.

The implementation uses a convolutional neural network to process visual observations and estimate action-value functions.

Key Components

Component	Description
Environment	Creates and preprocesses the Mario game environment.
DQN Network	Uses convolutional layers to estimate Q-values.
Replay Memory	Stores and samples past experiences.
Mario Agent	Implements epsilon-greedy action selection.
Configuration	Stores environment and training parameters.
Training Module	Initializes networks, optimizer, and replay memory.

Learning Approach

The project explores the following reinforcement learning concepts:

* Experience Replay: Randomly samples stored experiences to improve learning stability.
* Target Network: Maintains a separate network for target-value estimation.
* Epsilon-Greedy Strategy: Balances exploration and exploitation.
* Double DQN: Uses the online network to select the next action and the target network to evaluate it.

Technology Stack

* Python 3.11
* PyTorch
* Torchvision
* OpenAI Gym
* Gym Super Mario Bros.
* NES-Py
* NumPy
* PyYAML

Project Status

Core components have been implemented and tested individually. The Double DQN optimization and complete training pipeline are under development.
