import torch
import pygame
from torch.distributions import Categorical
import time
from maze_environment import Maze


import torch
import pygame
import time
from torch.distributions import Categorical

from maze_environment import Maze
from maze_grpo import GRPONetwor

def test_maze_agent(policy, maze_env, num_episodes=5, render_delay=0.1):
    """
    Test the trained maze agent with visualization
    
    Args:
        policy (GRPONetwork): Trained policy network
        maze_env (Maze): Maze environment
        num_episodes (int): Number of episodes to test
        render_delay (float): Delay between frames for visualization
    """
    # Initialize Pygame display
    pygame.init()
    SCALE_FACTOR = 3
    WINSIZE = (maze_env.w * 16 * SCALE_FACTOR, maze_env.h * 16 * SCALE_FACTOR)
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Maze GRPO Testing')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Set policy to evaluation mode
    policy.eval()
    
    # Determine device
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else
              "cpu")
    policy = policy.to(device)
    
    # Test episodes
    for episode in range(num_episodes):
        current_state = maze_env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Clear screen
            screen.fill((0, 0, 0))
            
            # Render current state
            maze_env.draw(screen)
            
            # Display episode info
            episode_text = font.render(f'Episode: {episode + 1}/{num_episodes}', True, (255, 255, 255))
            steps_text = font.render(f'Steps: {steps}', True, (255, 255, 255))
            reward_text = font.render(f'Total Reward: {total_reward:.1f}', True, (255, 255, 255))
            
            screen.blit(episode_text, (10, 10))
            screen.blit(steps_text, (10, 40))
            screen.blit(reward_text, (10, 70))
            
            pygame.display.flip()
            time.sleep(render_delay)  # Add delay to make visualization easier to follow
            
            # Get action from policy
            state_tensor = torch.FloatTensor(current_state).to(device)
            with torch.no_grad():
                action_logits = policy(state_tensor)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
            
            # Take step in environment
            next_state, reward, done, _ = maze_env.step(action.item())
            
            total_reward += reward
            steps += 1
            current_state = next_state
            
            # Optional: Break if max steps reached to prevent infinite loop
            if steps >= 1000:
                break
        
        # Print episode summary
        print(f"Episode {episode + 1} finished after {steps} steps with reward {total_reward:.2f}")
        time.sleep(1)  # Pause briefly between episodes
    
    # Cleanup
    pygame.quit()

def main():
    """Main function to load the best model and test the agent"""
    # Create maze environment
    maze_env = Maze(level=0)  # Start with smallest maze
    
    # Determine device
    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else
              "cpu")
    
    # Initialize network
    policy = GRPONetwork(obs_dim=4, act_dim=4).to(device)
    
    # Load the best model
    try:
        # Try loading from the last iteration's best model
        checkpoint = torch.load('saved_models/best_model_iter_2.pt')
        policy.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model from last iteration")
    except FileNotFoundError:
        try:
            # Fallback to final model
            checkpoint = torch.load('saved_models/final_model.pt')
            policy.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded final model")
        except FileNotFoundError:
            print("No saved model found. Please train the agent first.")
            return
    
    # Test the agent
    test_maze_agent(policy, maze_env, num_episodes=5, render_delay=0.1)

if __name__ == "__main__":
    main()