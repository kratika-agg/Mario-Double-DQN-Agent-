# Mario Double DQN Agent 🍄

A Deep Reinforcement Learning project that trains an agent to play Super Mario Bros using **Double Deep Q-Network (Double DQN)**.

The agent learns to select actions from visual game observations using a CNN-based Q-network, experience replay, and a target network.

## Project Overview

This project implements a Double DQN agent for the Super Mario Bros environment.

Key components:
- **Deep Q-Network (DQN):** Predicts Q-values for available actions.
- **Double DQN:** Uses the policy network to select the next action and the target network to evaluate it.
- **Experience Replay:** Stores and samples past transitions for training.
- **Target Network:** Stabilizes Q-value learning.
- **Epsilon-Greedy Policy:** Balances exploration and exploitation.
- **Training Metrics:** Tracks episode rewards, average rewards, loss, epsilon, and global steps.

## Project Structure

```text
Mario_Double_DQN_Agent/
│
├── environment.py
├── dqn.py
├── agent.py
├── experience_replay.py
│
├── train.py
├── play.py
├── parameters.yaml
│
├── plot_training.py
│
├── test_agent.py
├── test_config.py
├── test_mario.py
├── test_optimize.py
├── test_replay.py
│
├── README.md
└── .gitignore
