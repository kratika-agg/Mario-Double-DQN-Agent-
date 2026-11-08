Mario Double DQN Agent

A reinforcement learning implementation of a Mario-playing agent using Double Deep Q-Networks (Double DQN), PyTorch, and the Super Mario Bros. environment.

Overview

This project explores how an autonomous agent can learn to play Super Mario Bros. through reinforcement learning.

The agent processes visual observations from the game, selects actions using an epsilon-greedy policy, and learns from collected experiences.

A convolutional neural network estimates action-value functions, while the Double DQN approach separates next-action selection from next-action evaluation to help reduce overestimation bias.

Architecture

The project consists of the following components:

Module	Responsibility
environment.py	Environment creation, frame skipping, grayscale conversion, resizing, and frame stacking.
dqn.py	Convolutional neural network for Q-value estimation.
agent.py	Epsilon-greedy action selection.
experience_replay.py	Experience storage and random sampling.
parameters.yaml	Environment and training configuration.
train.py	Network, optimizer, and replay memory initialization, with training logic under development.

Reinforcement Learning Concepts

1. Deep Q-Network (DQN)

A convolutional neural network processes game observations and estimates Q-values for available actions.

2. Experience Replay

Transitions are stored in a replay buffer and sampled randomly to reduce correlations between consecutive experiences.

3. Target Network

A separate target network is maintained to provide more stable target-value estimates.

4. Epsilon-Greedy Strategy

The agent balances exploration and exploitation by selecting random actions with probability epsilon and predicted best actions otherwise.

5. Double DQN

The online network selects the next action, while the target network evaluates that action.

This separation helps reduce the overestimation of action values.

Environment Processing

The game observations are processed through the following pipeline:

1. Frame skipping
2. Grayscale conversion
3. Image resizing to 84 × 84
4. Stacking four consecutive frames

The resulting state provides visual and temporal information to the neural network.

Technology Stack

* Python 3.11
* PyTorch
* Torchvision
* NumPy
* OpenAI Gym
* Gym Super Mario Bros.
* NES-Py
* PyYAML

Setup and Installation

Clone the repository:

git clone https://github.com/kratika-agg/Mario-Double-DQN-Agent-.git
cd Mario-Double-DQN-Agent-

Create a virtual environment:

py -3.11 -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install the required packages:

pip install torch==2.2.2 torchvision==0.17.2 numpy==1.26.4
pip install gym==0.26.2 gym-super-mario-bros==7.4.0 nes-py==8.2.1
pip install tensordict==0.3.0 torchrl==0.3.0 pyyaml

Configuration

Training and environment parameters are maintained in parameters.yaml, including:

* Environment identifier
* Number of available actions
* Replay memory capacity
* Batch size
* Learning rate
* Discount factor
* Target network update frequency
* Exploration parameters
* Episode settings

Project Status

The environment, neural network, replay memory, agent, and configuration modules have been implemented and tested individually.

The training initialization is in place. Double DQN optimization and the complete training pipeline are currently under development.

Future Work

* Complete the Double DQN optimization function.
* Implement the complete training loop.
* Add model checkpointing.
* Evaluate agent performance across episodes.
* Analyze training rewards and learning progress.
