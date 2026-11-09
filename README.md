# Mario Double DQN Agent

A Deep Reinforcement Learning project that implements a **Double Deep Q-Network (Double DQN)** agent to learn gameplay strategies in the Super Mario Bros environment.

The project uses PyTorch and Convolutional Neural Networks to process visual game observations, estimate action values, and learn through interaction with the environment.

## Overview

The objective is to train an agent that learns to make decisions based on game states and rewards.

The implementation follows the Double DQN approach, combining:

- An online (policy) network for action selection.
- A target network for stable Q-value evaluation.
- Experience replay for sampling past interactions.
- Epsilon-greedy exploration for balancing exploration and exploitation.

## Technology Stack

- **Language:** Python 3.11
- **Deep Learning:** PyTorch
- **Computer Vision:** TorchVision
- **Numerical Computing:** NumPy
- **Environment:** Gym, Gym Super Mario Bros
- **Configuration:** PyYAML

## Architecture

```text
Mario Double DQN Agent
│
├── environment.py
│   ├── SkipFrame
│   ├── GrayScaleObservation
│   ├── ResizeObservation
│   └── FrameStack
│
├── dqn.py
│   └── CNN-based Q-Network
│
├── experience_replay.py
│   └── ReplayMemory
│
├── agent.py
│   └── Epsilon-Greedy Action Selection
│
├── parameters.yaml
│   └── Hyperparameter Configuration
│
├── train.py
│   ├── Policy Network
│   ├── Target Network
│   ├── Optimizer
│   └── Double DQN Optimization
│
└── Tests
    ├── test_mario.py
    ├── test_replay.py
    ├── test_agent.py
    ├── test_config.py
    └── test_optimize.py
```

## Environment Preprocessing

Raw game observations are transformed before being passed to the neural network.

The preprocessing pipeline includes:

1. **Action Space Reduction:** Restricts the agent to moving right and jumping right.
2. **Frame Skipping:** Repeats actions across multiple environment frames.
3. **Grayscale Conversion:** Reduces the observation to a single color channel.
4. **Resizing:** Converts observations to 84 × 84.
5. **Frame Stacking:** Combines four consecutive frames to capture temporal information.

The resulting observation represents four stacked grayscale frames.

## Double DQN Implementation

### 1. Online and Target Networks

Two instances of the same CNN architecture are initialized:

- **Policy Network:** Predicts Q-values and selects actions.
- **Target Network:** Evaluates selected next-state actions to calculate stable learning targets.

Initially, the target network receives the policy network's weights.

### 2. Experience Replay

Transitions are stored in replay memory:

\[
(s_t,a_t,r_t,s_{t+1},done)
\]

Random mini-batches are sampled from memory to reduce correlation between consecutive experiences.

### 3. Double DQN Optimization

The optimization process follows these steps:

**Step 1: Predict current Q-values**

The policy network predicts Q-values for the current states.

**Step 2: Select the executed action's Q-value**

Only the Q-value corresponding to the action actually performed is extracted using `gather()`.

**Step 3: Identify non-terminal transitions**

Terminal transitions do not require future-state value estimation.

**Step 4: Select next actions using the policy network**

\[
a^* = \arg\max_a Q_{\text{policy}}(s',a)
\]

**Step 5: Evaluate selected actions using the target network**

\[
Q_{\text{target}}(s',a^*)
\]

This separation between action selection and action evaluation is the central idea behind Double DQN.

**Step 6: Calculate the Bellman target**

\[
y = r + \gamma Q_{\text{target}}(s',a^*)
\]

For terminal transitions:

\[
y = r
\]

**Step 7: Calculate loss and update the policy network**

The implementation uses Smooth L1 Loss, backpropagation, gradient clipping, and the Adam optimizer.

## Hyperparameters

The current configuration is maintained in `parameters.yaml`.

| Parameter | Value |
|---|---:|
| Replay Memory Capacity | 100,000 |
| Batch Size | 32 |
| Learning Rate | 0.00025 |
| Discount Factor (Gamma) | 0.99 |
| Target Update Interval | 1,000 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.01 |
| Epsilon Decay | 0.995 |
| Planned Episodes | 10,000 |

These are configured values; actual training and evaluation are still in progress.

## Testing

The project includes individual tests for its main components.

The optimization test currently verifies:

- Replay memory can supply a mini-batch.
- The optimization function calculates a finite loss.
- Backpropagation executes successfully.
- Policy network weights are updated.

Run the optimization test using:

```bash
python test_optimize.py
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/kratika-agg/Mario-Double-DQN-Agent-.git
cd Mario-Double-DQN-Agent-
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

### 4. Install dependencies

Install the compatible package versions used by the project:

```bash
pip install torch==2.2.2 torchvision==0.17.2 numpy==1.26.4
pip install gym==0.26.2 gym-super-mario-bros==7.4.0
pip install nes-py==8.2.1 tensordict==0.3.0 torchrl==0.3.0 PyYAML
```

Note: The PyTorch installation above uses the CPU build. GPU-enabled training requires a compatible PyTorch build and CUDA setup.

## Development Roadmap

- [x] Environment creation and preprocessing
- [x] CNN architecture
- [x] Experience replay
- [x] Epsilon-greedy action selection
- [x] Online and target network initialization
- [x] Double DQN optimization implementation
- [x] Initial optimization test
- [ ] Terminal-transition testing
- [ ] Complete episode-based training loop
- [ ] Periodic target network synchronization
- [ ] Epsilon decay during training
- [ ] Model saving and loading
- [ ] Training performance evaluation
- [ ] Gameplay visualization

## Current Status

**Optimization implemented and initial test passed. Training pipeline under development.**

The next milestone is integrating the environment, agent, replay memory, and optimization function into a complete training loop.

---

**Project:** Mario Double DQN Agent  
**Status:** Under Development