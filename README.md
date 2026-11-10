# Mario Double DQN Agent

An AI agent that learns to play **Super Mario Bros.** using Deep Reinforcement Learning and the **Double Deep Q-Network (Double DQN)** algorithm.

The project uses PyTorch to train an agent that observes the game environment, selects actions, stores experiences, and improves its decision-making through repeated interactions.

## Project Overview

The agent uses a CNN-based Deep Q-Network to estimate Q-values for possible actions based on the current game state.

Double DQN improves upon standard DQN by separating action selection and action evaluation:

- **Policy Network:** Selects the best next action.
- **Target Network:** Evaluates the selected action.
- **Experience Replay:** Stores and samples past transitions for training.
- **Epsilon-Greedy Strategy:** Balances exploration and exploitation.

## Key Features

- Double DQN implementation using PyTorch
- Experience Replay Memory
- Separate Policy and Target Networks
- Epsilon-Greedy action selection
- Frame skipping and frame stacking
- Grayscale image processing and resizing
- Episode-based training
- Training reward and loss tracking
- CSV-based training metrics
- Reward and loss visualization
- Gameplay evaluation using a trained checkpoint

## Project Structure

```text
Mario_Double_DQN_Agent/
│
├── environment.py          # Mario environment and preprocessing
├── dqn.py                  # Deep Q-Network architecture
├── experience_replay.py    # Experience Replay Memory
├── agent.py                # Agent and action selection
├── train.py                # Double DQN training pipeline
├── play.py                 # Evaluate the trained agent
├── plot_training.py        # Plot training rewards and loss
├── parameters.yaml         # Hyperparameters and configurations
│
├── test_mario.py           # Environment testing
├── test_replay.py          # Replay memory testing
├── test_agent.py           # Agent testing
├── test_config.py          # Configuration testing
├── test_optimize.py        # Optimization testing
│
├── README.md
└── .gitignore
```

## Environment and State Preprocessing

The Mario environment is created using `gym-super-mario-bros`.

The observation preprocessing pipeline includes:

1. **Joypad Space:** Restricts the available actions to moving right and moving right with the jump button.
2. **Frame Skipping:** Repeats each selected action for 4 frames.
3. **Grayscale Conversion:** Converts RGB observations into grayscale images.
4. **Image Resizing:** Resizes observations to 84 × 84 pixels.
5. **Frame Stacking:** Stacks 4 consecutive frames to provide temporal information.

The final state representation has the shape:

```text
(4, 84, 84)
```

## Double DQN Training

The training process follows these steps:

1. Initialize the Mario environment, policy network, target network, and replay memory.
2. Observe the current state.
3. Select an action using the epsilon-greedy strategy.
4. Execute the action and observe the next state and reward.
5. Store the transition in replay memory.
6. Sample a mini-batch of transitions.
7. Calculate current Q-values using the policy network.
8. Select next actions using the policy network.
9. Evaluate selected next actions using the target network.
10. Calculate the target Q-values and optimize the policy network.
11. Periodically synchronize the target network.
12. Repeat for multiple episodes.

## Training Metrics

The training script records the following metrics after every episode:

| Metric | Description |
|---|---|
| Episode | Current training episode |
| Reward | Total reward obtained |
| Average Reward (100) | Average reward over the latest 100 episodes |
| Average Loss | Average optimization loss during the episode |
| Epsilon | Current exploration rate |
| Global Steps | Total environment steps |

Metrics are stored in:

```text
training_metrics.csv
```

## Visualizing Training Performance

Run the following command after training:

```bash
python plot_training.py
```

The script generates two visualizations:

- `reward_curve.png` – Episode rewards and average reward over 100 episodes.
- `training_loss.png` – Average training loss across episodes.

These graphs help monitor training progress and optimization behaviour.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kratika-agg/Mario-Double-DQN-Agent-.git
cd Mario-Double-DQN-Agent-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

Install the compatible project dependencies in your virtual environment. The current tested setup uses:

- Python 3.11
- PyTorch 2.2.2 (CPU)
- Torchvision 0.17.2
- NumPy 1.26.4
- Gym
- Gym Super Mario Bros.
- Matplotlib
- PyYAML

For this legacy Mario environment, NumPy 1.26.4 is used for compatibility.

### 5. Run the training script

```bash
python train.py
```

Training parameters can be configured in `parameters.yaml`.

### 6. Evaluate the trained agent

```bash
python play.py
```

The evaluation script loads the saved model checkpoint and runs the agent using its learned policy.

### 7. Generate training graphs

```bash
python plot_training.py
```

## Current Implementation Status

- [x] Mario environment setup
- [x] State preprocessing and frame stacking
- [x] CNN-based DQN architecture
- [x] Experience Replay Memory
- [x] Double DQN optimization
- [x] Episode-based training pipeline
- [x] Target network synchronization
- [x] Model checkpoint saving
- [x] Training metrics in CSV format
- [x] Reward and loss visualization
- [x] Basic gameplay evaluation pipeline
- [ ] Extended training and performance evaluation

## Current Testing

The training pipeline has successfully completed a smoke test using:

- Episodes: 2
- Maximum steps per episode: 40
- Total environment steps: 80

The test verified that the environment, training loop, model saving, CSV logging, and graph generation work together.

This is an initial pipeline validation, not evidence of a fully trained or converged agent.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- OpenAI Gym
- Gym Super Mario Bros.
- Matplotlib
- PyYAML
- Git and GitHub

## Future Improvements

- Extended training and reward analysis
- Evaluate agent performance over multiple episodes
- Improve training efficiency
- Analyze the effect of different hyperparameter configurations

---

**Project:** Mario Double DQN Agent  
**Focus:** Deep Reinforcement Learning | Computer Vision | PyTorch
