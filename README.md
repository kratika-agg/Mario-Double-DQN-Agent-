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