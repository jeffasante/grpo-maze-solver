# GRPO Maze Solver

A reinforcement learning agent that learns to solve mazes using **Group Relative Policy Optimization (GRPO)**. Watch an AI learn to navigate through mazes in real-time!

## What is this?

This project demonstrates how an AI agent can learn to solve mazes through trial and error using GRPO—a variant of Proximal Policy Optimization (PPO) that leverages group-relative advantage estimation and iterative policy updates. The agent begins with no knowledge of the maze, collects trajectories, and updates its policy based on group comparisons. Additionally, a reward model is trained using a replay mechanism to progressively refine reward estimations.

### How it Works

1. **Observation and Action:**  
   The agent (displayed as a red square) observes its current position and the target location, then selects an action (up, down, left, or right) using its policy network.

2. **Trajectory Collection:**  
   Multiple trajectories are collected in each episode. Each trajectory's cumulative reward is computed based on:
   - A positive reward (e.g., 1000) when the goal is reached.
   - Negative rewards for hitting walls or not making progress.
   - A small bonus per step to encourage faster solutions.

3. **Group Relative Advantage Estimation:**  
   After collecting a batch of trajectories, the agent computes relative advantages by comparing each trajectory's reward to the group average. This group-relative advantage helps guide the policy update.

4. **Iterative GRPO Update (Equation 21):**  
   The policy is updated over multiple GRPO iterations per trajectory update. The update uses:
   - A surrogate loss similar to PPO with clipping.
   - A KL divergence penalty against a reference policy to maintain stability.
   - Configurable hyperparameters such as the clipping parameter (\( \epsilon \)), KL coefficient (\( \beta \)), and the number of GRPO iterations (\( \mu \)).

5. **Reward Model Update:**  
   The reward model is continuously updated via a replay mechanism using the experiences stored in an experience buffer. An MSE loss is used (which can be replaced with a contrastive loss) to refine reward predictions over time.

## Key Features

- **Real-time Visualization:**  
  Watch the agent learn in real-time using PyGame.

- **Iterative GRPO Updates:**  
  The agent performs multiple GRPO iterations per trajectory update to efficiently optimize its policy based on group-relative advantages.

- **Reward Model Replay:**  
  The reward model is updated using a replay buffer to ensure robust reward estimation.

- **Configurable Hyperparameters:**  
  Easily adjust key parameters such as the clipping parameter (\( \epsilon \)), KL coefficient (\( \beta \)), and GRPO iteration count (\( \mu \)).

- **Performance Monitoring:**  
  Training progress is logged via TensorBoard, showing metrics such as average reward, best reward, success rate, and steps per episode.

## Requirements

```bash
python >= 3.8
torch
pygame
numpy
tensorboard
```

## Installation

1. Clone the repository:
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

To train a new agent with GRPO:
```bash
python maze_grpo.py
```

The script will automatically detect the best available device (CUDA, MPS, or CPU). On Apple Silicon Macs, it will use MPS for accelerated training.

**Note:**  
The training script now accepts a new parameter `grpo_iterations` (default is 3) that controls how many GRPO updates are performed per trajectory update. You can modify this (and other hyperparameters such as `epsilon` and `beta`) directly in the code.

### Sample Training Output

An example of the training output might look like:

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
...
New best model saved! Reward: 1227.65
```

### Testing

To test a trained agent:
```bash
python test_maze.py
```

## How it Works (The Technical Bits)

### The Brain (Neural Network)

- **Policy Network (GRPONetwork):**  
  Receives the state input \([x, y, target_x, target_y]\) and outputs action probabilities for each movement direction.

- **Reward Network (GRPORewardNetwork):**  
  Combines the state and one-hot encoded action to predict a reward value.

### The GRPO Objective

The GRPO update follows a PPO-like objective with an iterative update loop:
```python
ratio = exp(new_log_probs - old_log_probs)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
policy_loss = -min(surr1, surr2).mean()
kl_loss = beta * KL(π_θ || π_ref)
total_loss = policy_loss + kl_loss
```
This objective is iteratively optimized (controlled by `grpo_iterations`) to better align the policy with group-relative advantages.

### Reward Model Training

The reward model is updated using a replay mechanism:
- Experiences (state, action, reward) are stored in an **ExperienceBuffer**.
- When the buffer is sufficiently large, a batch is sampled and used to update the reward model via MSE loss (or another loss function as desired).

### Hyperparameter Configuration

Key parameters include:
```python
num_episodes    = 5000    # Total training episodes
group_size      = 64      # Number of trajectories per update
grpo_iterations = 3       # GRPO update iterations (μ)
learning_rate   = 1e-4    # Learning rate for policy updates
gamma           = 0.99    # Discount factor
epsilon         = 0.2     # Clipping parameter for PPO-style loss
beta            = 0.04    # KL divergence coefficient
```
You can adjust these values in the source code to suit your training requirements.

## Monitoring Progress

Training metrics are logged via TensorBoard. To view the logs:
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

**Author:** Jeffrey Asante  
**Date:** 2025-02-08  
**GitHub:** [github.com/jeffasante](https://github.com/jeffasante)

---
