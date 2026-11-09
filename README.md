# 🍄 Mario Double DQN Agent

An AI agent that learns to play **Super Mario Bros.** using Deep Reinforcement Learning and the Double Deep Q-Network (Double DQN) algorithm.

The project implements a complete reinforcement learning pipeline, including environment preprocessing, experience replay, epsilon-greedy exploration, neural network optimization, and gameplay evaluation.

## 🚀 Project Overview

The objective is to train an intelligent agent that learns to navigate the Super Mario Bros. environment by interacting with it, collecting rewards, and improving its action-selection policy.

The agent uses visual observations of the game as input and predicts Q-values for available actions.

## 🧠 Algorithm: Double DQN

This project uses Double DQN to reduce the overestimation of action values commonly associated with traditional DQN.

It uses two neural networks:

- **Policy Network (Online Network):** Selects the best next action.
- **Target Network:** Evaluates the selected action to calculate target Q-values.

The networks are periodically synchronized during training.

## 🏗️ Project Architecture

```text
Mario_Double_DQN_Agent/
│
├── environment.py
├── dqn.py
├── agent.py
├── experience_replay.py
├── train.py
├── play.py
├── parameters.yaml
├── README.md
└── .gitignore
```

### File Descriptions

| File | Description |
|---|---|
| `environment.py` | Creates and preprocesses the Mario environment |
| `dqn.py` | Defines the Deep Q-Network architecture |
| `agent.py` | Implements epsilon-greedy action selection |
| `experience_replay.py` | Stores and samples past experiences |
| `train.py` | Implements the Double DQN training pipeline |
| `play.py` | Loads the saved model and evaluates gameplay |
| `parameters.yaml` | Stores configurable training parameters |

## ⚙️ Technologies Used

- Python
- PyTorch
- OpenAI Gym
- Gym Super Mario Bros
- NumPy
- YAML
- Torchvision

## 🔄 Environment Preprocessing

The game environment is processed to make observations suitable for the neural network.

The preprocessing pipeline includes:

1. Frame skipping to reduce computational overhead.
2. Grayscale conversion to simplify visual input.
3. Resizing frames to 84 × 84 pixels.
4. Frame stacking to provide temporal information.

The resulting observation contains four stacked grayscale frames.

## 🧠 Deep Q-Network Architecture

The neural network uses a convolutional architecture to extract visual features from game observations.

Architecture:

- Convolutional layer: 32 filters, kernel size 8, stride 4.
- Convolutional layer: 64 filters, kernel size 4, stride 2.
- Convolutional layer: 64 filters, kernel size 3, stride 1.
- Fully connected layer: 512 neurons.
- Output layer: Q-values for available actions.

ReLU activation functions are used between layers.

## 🎮 Action Selection

The agent uses an epsilon-greedy strategy:

- With probability ε, it selects a random action to explore.
- Otherwise, it selects the action with the highest predicted Q-value.

Epsilon gradually decreases during training to shift from exploration towards exploitation.

## 🗃️ Experience Replay

The replay memory stores transitions containing:

- Current state
- Selected action
- Next state
- Reward
- Terminal flag

Random batches of experiences are sampled during optimization to improve learning stability and reduce correlations between consecutive observations.

## ⚡ Double DQN Optimization

The optimization pipeline includes:

1. Sampling a batch from replay memory.
2. Predicting current Q-values using the policy network.
3. Selecting next actions using the policy network.
4. Evaluating selected next actions using the target network.
5. Calculating target Q-values using the Bellman equation.
6. Computing Smooth L1 loss.
7. Performing backpropagation.
8. Applying gradient clipping.
9. Updating policy network weights.

The target network is periodically synchronized with the policy network.

## 🛠️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kratika-agg/Mario-Double-DQN-Agent-.git
cd Mario-Double-DQN-Agent-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

Install the required packages:

```bash
pip install torch torchvision gym gym-super-mario-bros nes-py numpy pyyaml
```

Ensure your installed package versions are compatible with the environment implementation.

## 🏋️ Training the Agent

Configure training parameters in `parameters.yaml`.

Important parameters include:

| Parameter | Description |
|---|---|
| `memory_size` | Replay memory capacity |
| `batch_size` | Number of sampled experiences |
| `learning_rate` | Optimizer learning rate |
| `gamma` | Discount factor |
| `target_update` | Target network synchronization interval |
| `epsilon_start` | Initial exploration rate |
| `epsilon_end` | Minimum exploration rate |
| `epsilon_decay` | Exploration decay factor |
| `num_episodes` | Number of training episodes |
| `max_steps_per_episode` | Maximum steps per episode |

Start training:

```bash
python train.py
```

The training script reports episode rewards, epsilon values, total steps, and average loss.

After training, the policy network weights are saved to:

```text
mario_double_dqn.pth
```

## 🎯 Gameplay Evaluation

To load the saved checkpoint and run the agent:

```bash
python play.py
```

The evaluation script:

1. Loads the saved policy network weights.
2. Initializes the Mario environment.
3. Selects actions greedily using the highest predicted Q-value.
4. Runs a gameplay episode.
5. Reports total reward and number of steps.

The evaluation script uses epsilon = 0, meaning the agent does not perform random exploration.

**Note:** The current checkpoint was generated during a short smoke test. Meaningful gameplay performance requires extended training.

## 📊 Current Implementation Status

- [x] Mario environment setup
- [x] Grayscale and resize preprocessing
- [x] Frame stacking
- [x] Deep Q-Network architecture
- [x] Policy and target networks
- [x] Epsilon-greedy action selection
- [x] Experience replay memory
- [x] Double DQN optimization
- [x] Terminal-state handling
- [x] Episode-based training loop
- [x] Model checkpoint saving
- [x] Gameplay evaluation script
- [ ] Extended model training
- [ ] Training performance analysis
- [ ] Evaluation across multiple episodes

## 🔮 Future Improvements

- Train the agent over extended episodes.
- Track and visualize episode rewards.
- Analyze training loss and learning progress.
- Evaluate performance across multiple episodes.
- Improve training efficiency and checkpoint management.

## 👩‍💻 Author

**Kratika Agrawal**

GitHub: [kratika-agg](https://github.com/kratika-agg)

---

*Built as a hands-on Deep Reinforcement Learning project to understand and implement Double DQN using PyTorch.*