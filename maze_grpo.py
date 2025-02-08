''' 
Author: Jeffrey Asante
Date: 2025-02-8
Description: Implementation of Group Relative Policy Optimization (GRPO) for training an agent to solve a maze environment.
Github: github.com/jeffasante
'''

import torch
import torch.nn as nn
import torch.nn.functional as F  # For loss functions
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import pygame

'''
ExperienceBuffer: For storing experiences for reward model training
GroupBuffer: For storing and calculating relative advantages between groups of trajectories
GRPORewardNetwork: Neural network for predicting rewards in GRPO
GRPONetwork: Neural network for policy in GRPO
'''

# Group Buffer for storing policies
class GroupBuffer:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.policies = []
        self.returns = []
        
    def add(self, policy, avg_return):
        if len(self.policies) >= self.max_size:
            self.policies.pop(0)
            self.returns.pop(0)
        self.policies.append(policy)
        self.returns.append(avg_return)
    
    def calculate_relative_advantage(self, rewards):
        """Calculate advantages relative to group performance"""
        if not rewards:
            return []
        group_mean = np.mean(rewards)
        group_std = np.std(rewards) + 1e-8
        return (np.array(rewards) - group_mean) / group_std
    
    def mean_return(self):
        return sum(self.returns) / len(self.returns) if self.returns else 0

# GRPO Network (Policy)
class GRPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # For maze: obs_dim will be 4 (x, y, target_x, target_y)
        # act_dim will be 4 (up, down, left, right)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        
    def forward(self, x):
        return self.actor(x)

# Reward network for GRPO
class GRPORewardNetwork(nn.Module):
    """Neural network for predicting rewards in GRPO"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim  # Store action dimension
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        # Create one-hot tensor on the same device as input tensors
        action_onehot = torch.zeros(action.size(0), self.act_dim, device=action.device)
        action_onehot.scatter_(1, action.unsqueeze(1), 1)
        # Concatenate and pass through network
        x = torch.cat([state, action_onehot], dim=1)
        return self.network(x).squeeze(-1)

# Reward model training buffer
class ExperienceBuffer:
    """Buffer for storing trajectories for reward model training"""
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        
    def add(self, state, action, reward):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        states = torch.stack([self.states[i] for i in indices])
        actions = torch.stack([self.actions[i] for i in indices])
        rewards = torch.FloatTensor([self.rewards[i] for i in indices])
        return states, actions, rewards
        
    def __len__(self):
        return len(self.states)

def calculate_kl_divergence(policy, reference_policy, states):
    with torch.no_grad():
        ref_logits = reference_policy(states)
    new_logits = policy(states)
    kl_div = torch.distributions.kl.kl_divergence(
        Categorical(logits=new_logits), 
        Categorical(logits=ref_logits)
    )
    return kl_div

# Train the agent using GRPO
def train_maze_grpo(maze_env, num_episodes=5000, group_size=64, grpo_iterations=3):
    """
    Train an agent using Group Relative Policy Optimization (GRPO)
    with proper episode completion handling and improved trajectory collection.
    
    Args:
        maze_env: Maze environment instance.
        num_episodes: Maximum number of episodes to train.
        group_size: Number of trajectories to collect before updating policy.
        grpo_iterations: Number of GRPO iterations (μ) per trajectory update.
    """
    # Initialize Pygame for visualization
    pygame.init()
    SCALE_FACTOR = 3
    WINSIZE = (maze_env.w * 16 * SCALE_FACTOR, maze_env.h * 16 * SCALE_FACTOR)
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Maze GRPO Training')
    clock = pygame.time.Clock()
    
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir="./runs/maze_training")
    
    # Initialize GRPO components
    obs_dim = 4  # State dimensions: (x, y, target_x, target_y)
    act_dim = 4  # Action space: (up, down, left, right)
    
    # Set up device (GPU/MPS if available, else CPU)
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else
              "cpu")
    print(f"Using device: {device}")
    
    # Initialize neural networks
    policy = GRPONetwork(obs_dim, act_dim).to(device)
    reference_policy = GRPONetwork(obs_dim, act_dim).to(device)
    reward_network = GRPORewardNetwork(obs_dim, act_dim).to(device)
    reference_policy.load_state_dict(policy.state_dict())

    # Setup optimizers and buffers
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    reward_optimizer = optim.Adam(reward_network.parameters(), lr=2e-5)
    group_buffer = GroupBuffer(max_size=5)
    experience_buffer = ExperienceBuffer(max_size=100000)
    
    # Training hyperparameters
    gamma = 0.99        # Discount factor for future rewards
    epsilon = 0.2       # PPO clipping parameter
    beta = 0.04         # KL divergence coefficient
    max_steps = 1000    # Maximum steps per episode
    num_iterations = 3  # Number of major iterations
    episodes_per_iter = num_episodes // num_iterations

    # Setup tracking variables
    best_reward = float('-inf')
    episode_rewards = []
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize font for display
    font = pygame.font.Font(None, 36)
    
    # Training statistics
    successful_episodes = 0
    total_steps = 0
    
    # Main training loop over iterations
    for iteration in range(num_iterations):
        print(f"\nStarting Iteration {iteration + 1}/{num_iterations}")
        
        # Set reference policy to current policy at start of iteration
        reference_policy.load_state_dict(policy.state_dict())
        
        episode = iteration * episodes_per_iter
        running = True
        
        while running and episode < (iteration + 1) * episodes_per_iter:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                    
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")
            
            # Initialize group collection variables
            group_states = []
            group_actions = []
            group_rewards = []
            group_log_probs = []
            group_total_rewards = []
            group_successes = 0
            
            # Collect group of trajectories
            valid_trajectories = 0
            group_attempts = 0
            max_group_attempts = group_size * 2  # Allow some retry attempts
            
            while valid_trajectories < group_size and group_attempts < max_group_attempts:
                # Initialize trajectory variables
                states, actions, log_probs, rewards = [], [], [], []
                state = maze_env.reset()
                total_reward = 0
                steps = 0
                episode_complete = False
                
                # Single trajectory collection loop
                while not episode_complete and steps < max_steps:
                    # Clear screen and draw current state
                    screen.fill((0, 0, 0))
                    maze_env.draw(screen)
                    
                    # Display current status
                    texts = [
                        f'Episode: {episode + 1}/{num_episodes}',
                        f'Valid Trajectories: {valid_trajectories}/{group_size}',
                        f'Steps: {steps}',
                        f'Total Reward: {total_reward:.1f}',
                        f'Best Reward: {best_reward:.1f}',
                        f'Success Rate: {(successful_episodes/max(1, episode)):.2%}'
                    ]
                    
                    # Render status texts
                    for i, text in enumerate(texts):
                        text_surface = font.render(text, True, (255, 255, 255))
                        screen.blit(text_surface, (10, 10 + i * 30))
                    
                    pygame.display.flip()
                    clock.tick(60)

                    # Convert state to tensor and get action from policy
                    state_tensor = torch.FloatTensor(state).to(device)
                    with torch.no_grad():
                        action_logits = policy(state_tensor)
                        dist = Categorical(logits=action_logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    
                    # Take step in environment
                    next_state, step_reward, done, _ = maze_env.step(action.item())
                    
                    # Get reward from reward network
                    with torch.no_grad():
                        model_reward = reward_network(state_tensor, action.unsqueeze(0))
                    
                    # Store experience for reward model training
                    experience_buffer.add(state_tensor, action, step_reward)
                    
                    # Store transition
                    states.append(state)
                    actions.append(action.item())
                    log_probs.append(log_prob.item())
                    
                    # Check win condition and update reward
                    if maze_env.check_win():
                        reward = 1000.0  # Ensure win reward is given
                        rewards.append(reward)
                        total_reward += reward
                        print(f"  Trajectory {valid_trajectories + 1}: Success! Steps = {steps}, Final Reward = {total_reward:.2f}")
                        episode_complete = True
                        group_successes += 1
                    else:
                        # Use reward from model and add step bonus
                        step_reward = model_reward.item() + 0.1
                        rewards.append(step_reward)
                        total_reward += step_reward
                        
                        # Check other completion conditions
                        if steps >= max_steps:
                            print(f"  Trajectory {valid_trajectories + 1}: Max steps reached. Reward = {total_reward:.2f}")
                            episode_complete = True
                        elif done:
                            collision_penalty = -0.5
                            rewards.append(collision_penalty)
                            total_reward += collision_penalty
                            episode_complete = True
                    
                    state = next_state
                    steps += 1
                
                # Store trajectory if complete
                if episode_complete:
                    group_states.append(states)
                    group_actions.append(actions)
                    group_rewards.append(rewards)
                    group_log_probs.append(log_probs)
                    group_total_rewards.append(total_reward)
                    valid_trajectories += 1
                    total_steps += steps
                
                group_attempts += 1
            
            if not running:
                break
                
            # Only proceed with updates if we have collected enough trajectories
            if valid_trajectories > 0:
                # Update reward network with experiences (using MSE loss here)
                if len(experience_buffer) > 1000:  # Minimum size before training
                    states_batch, actions_batch, rewards_batch = experience_buffer.sample(256)
                    predicted_rewards = reward_network(states_batch, actions_batch)
                    reward_loss = F.mse_loss(predicted_rewards, rewards_batch)
                    
                    reward_optimizer.zero_grad()
                    reward_loss.backward()
                    reward_optimizer.step()
                
                # Calculate group-relative advantages
                relative_advantages = group_buffer.calculate_relative_advantage(group_total_rewards)
                
                # Policy update loop with GRPO iterations
                for trajectory_idx in range(valid_trajectories):
                    # Convert trajectory data to tensors
                    states_tensor = torch.FloatTensor(group_states[trajectory_idx]).to(device)
                    actions_tensor = torch.LongTensor(group_actions[trajectory_idx]).to(device)
                    old_log_probs_tensor = torch.FloatTensor(group_log_probs[trajectory_idx]).to(device)
                    advantage_tensor = torch.FloatTensor([relative_advantages[trajectory_idx]] * len(states_tensor)).to(device)
                    
                    # Perform multiple GRPO iterations for this trajectory
                    for _ in range(grpo_iterations):
                        action_logits = policy(states_tensor)
                        dist = Categorical(logits=action_logits)
                        new_log_probs = dist.log_prob(actions_tensor)
                        
                        # Compute the ratio and surrogate loss
                        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                        surr1 = ratio * advantage_tensor
                        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage_tensor
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Compute KL divergence loss
                        kl_loss = beta * calculate_kl_divergence(policy, reference_policy, states_tensor).mean()
                        
                        # Total GRPO loss (to be minimized)
                        total_loss = policy_loss + kl_loss
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        
                        # Optional: Adaptive beta update based on KL divergence could be inserted here
                        # For example:
                        # if kl_loss.item() > 0.02: beta *= 1.1
                        # elif kl_loss.item() < 0.005: beta /= 1.1
                        
                # Update group buffer and track rewards
                avg_reward = np.mean(group_total_rewards)
                episode_rewards.append(avg_reward)
                group_buffer.add(policy.state_dict(), avg_reward)
                
                # Update success statistics
                successful_episodes += (group_successes > 0)
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save({
                        'iteration': iteration,
                        'episode': episode,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'reward_model_state_dict': reward_network.state_dict(),
                        'reward': avg_reward,
                        'steps': total_steps,
                        'successes': successful_episodes,
                    }, os.path.join(save_dir, f'best_model_iter_{iteration}.pt'))
                    print(f"\nNew best model saved! Reward: {avg_reward:.2f}")
                
                # Logging
                if episode % 10 == 0:
                    writer.add_scalar("Training/Average_Reward", avg_reward, episode)
                    writer.add_scalar("Training/Best_Reward", best_reward, episode)
                    writer.add_scalar("Training/Success_Rate", successful_episodes/(episode+1), episode)
                    writer.add_scalar("Training/Average_Steps", total_steps/(episode+1), episode)
                    writer.add_scalar("Training/Reward_Loss", reward_loss.item(), episode)
                    
                    avg_last_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    print(f"\nEpisode {episode + 1} Summary:")
                    print(f"  Iteration: {iteration + 1}/{num_iterations}")
                    print(f"  Average Reward: {avg_reward:.2f}")
                    print(f"  Best Reward: {best_reward:.2f}")
                    print(f"  Success Rate: {(successful_episodes/(episode+1)):.2%}")
                    print(f"  Average Steps per Episode: {(total_steps/(episode+1)):.1f}")
                    print(f"  Last 100 Episodes Average: {avg_last_100:.2f}")
                    print(f"  Reward Model Loss: {reward_loss.item():.4f}")
            
            # Update episode counter
            episode += 1
            pygame.time.wait(100)  # Reduced delay between episodes for faster training
        
    # Final cleanup
    pygame.quit()
    writer.close()
    
    # Save final models
    torch.save({
        'episode': episode,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward_model_state_dict': reward_network.state_dict(),
        'reward': avg_reward if 'avg_reward' in locals() else 0,
        'steps': total_steps,
        'successes': successful_episodes,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return policy, reward_network

if __name__ == '__main__':
    from maze_environment import Maze  

    # Create maze environment
    maze_env = Maze(level=0)  # Start with smallest maze

    # Train the agent
    policy, reward_network = train_maze_grpo(maze_env, num_episodes=5000)

    # Save the trained policy
    torch.save(policy.state_dict(), "maze_policy.pt")
