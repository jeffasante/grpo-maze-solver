# GRPO Maze Solver

A reinforcement learning agent that learns to solve mazes using **Group Relative Policy Optimization (GRPO)**. Watch an AI learn to navigate through mazes in real-time!

## What is this?

This project demonstrates how an AI agent can learn to solve mazes through trial and error. Using Group Relative Policy Optimization (GRPO)—a more efficient variant of PPO—the agent starts with no prior knowledge of the maze and gradually improves its navigation skills by learning from its experiences and comparing its performance against a group of previous attempts.

### How it Works

The agent (displayed as a red square) must reach the goal (green square). At each step, it:
1. Observes its current position and the goal location.
2. Decides which direction to move (up, down, left, or right).
3. Receives feedback (rewards) based on its actions.
4. Compares its performance with a group of previous trajectories.
5. Adjusts its strategy based on these group comparisons to improve future performance.

## Key Features

- **Real-time Visualization**: Watch the agent learn in real-time using PyGame.
- **Group Relative Learning**: Leverages group performance comparisons to derive more robust advantage estimates.
- **Reference Policy Tracking**: Maintains a reference policy for stable learning.
- **Adaptive Behavior**: The agent learns to avoid walls and optimize its path to the goal.
- **Performance Monitoring**: Training progress is logged and visualized with TensorBoard.

## Requirements

```bash
python >= 3.8
torch
pygame
numpy
tensorboard
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jeffasante/grpo-maze-solver
   cd grpo-maze-solver
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy pygame tensorboard
   ```

## Usage

### Training

To train a new agent:
```bash
python maze_grpo.py
```

The agent will automatically detect and use the best available device (CUDA, MPS, or CPU). On Apple Silicon Macs, it will use MPS (Metal Performance Shaders) for accelerated training.

### Sample Training Results

Below is an example of early training progress (first episode of iteration 1):

```
Using device: mps

Starting Iteration 1/3

Starting Episode 1/5000
  Trajectory 1: Success! Steps = 231, Final Reward = 1057.52
  Trajectory 2: Success! Steps = 214, Final Reward = 1055.43
  Trajectory 3: Success! Steps = 790, Final Reward = 1202.52
  Trajectory 4: Success! Steps = 367, Final Reward = 1095.95
  Trajectory 5: Success! Steps = 880, Final Reward = 1227.65
  Trajectory 6: Success! Steps = 737, Final Reward = 1187.62
```

*Note:* The rewards shown here are positive, reflecting a successful early convergence in some trajectories. As training continues over more episodes, you can expect further performance improvements and refined navigation behaviors.

### Testing

To test a trained agent:
```bash
python test_maze.py
```

## How it Works (The Technical Bits)

### The Brain (Neural Network)

The agent uses a single neural network—the policy network—to decide what action to take in each state. Unlike traditional PPO which uses both policy and value networks, GRPO achieves improved efficiency by employing group comparisons instead of a separate value network.

```python
Input: [current_x, current_y, goal_x, goal_y]
Output: [probability of each direction]
```

### The Math Behind Learning

#### Reward System
```python
R = {
    100  if reaches goal,
    -1   if hits wall,
    -0.1 × distance_to_goal  otherwise
}
```

#### Group Relative Advantage
Instead of relying on a value network, GRPO computes advantages relative to the performance of a group of trajectories:
```python
advantage = (reward - group_mean) / (group_std + 1e-8)
```

#### Policy Update
The policy update in GRPO involves a modified objective that incorporates both the PPO clipped loss and a KL divergence term to maintain stability:
```python
L = min(
    r_t(θ)A_t,
    clip(r_t(θ), 1-ε, 1+ε)A_t
) + β * KL(π_θ || π_ref)
```
where:
- `r_t(θ)` is the probability ratio,
- `A_t` is the group-relative advantage,
- `ε` is the clip range (typically 0.2),
- `β` is the KL divergence coefficient,
- `π_ref` is the reference policy.

### Group Buffer

The agent maintains a memory of its last 5 policies and their performance. This group buffer is used to:
- Calculate relative advantages for new experiences.
- Maintain stable learning via reference policy comparisons.
- Track the progress of learning over time.
- Help prevent catastrophic forgetting of good policies.

## Configuration

You can adjust the following key parameters:

```python
num_episodes = 5000    # Number of training episodes
learning_rate = 1e-4   # Learning rate for the policy network
gamma = 0.99         # Discount factor for future rewards
clip_epsilon = 0.2   # PPO clipping parameter
group_size = 64      # Number of trajectories per policy update
beta = 0.04          # KL divergence coefficient
```

## Monitoring Progress

Training progress is logged using TensorBoard. You can view:
- Average reward per episode.
- Group relative performance metrics.
- KL divergence and policy update statistics.

To view the logs, run:
```bash
tensorboard --logdir=runs/maze_training
```

## Credits and References

### Implementation References
- Maze environment based on: [FrankRuis's Maze Implementation](https://gist.github.com/FrankRuis/4bad6a988861f38cf53b86c185fc50c3)
- Visualization using PyGame.

### Academic References
- Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* – Introduces the GRPO algorithm.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv preprint arXiv:1707.06347.

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

**Author:** Jeffrey Oduro Asante  
**Date:** 2025-02-08  
**GitHub:** [github.com/jeffasante](https://github.com/jeffasante)

---
