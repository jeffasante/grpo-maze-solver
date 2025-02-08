import pygame
from random import choice
import os
from pygame.locals import *
import numpy as np

class Cell(pygame.sprite.Sprite):
    w, h = 16, 16
    
    def __init__(self, x, y, maze):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([self.w, self.h])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.x = x * self.w
        self.rect.y = y * self.h
        self.x = x
        self.y = y
        self.maze = maze
        self.nbs = [(x + nx, y + ny) for nx, ny in ((-2, 0), (0, -2), (2, 0), (0, 2))
                    if 0 <= x + nx < maze.w and 0 <= y + ny < maze.h]
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)

class Wall(Cell):
    def __init__(self, x, y, maze):
        super(Wall, self).__init__(x, y, maze)
        self.image.fill((0, 0, 0))
        self.type = 0

class Player(Cell):
    def __init__(self, x, y, maze):
        super(Player, self).__init__(x, y, maze)
        self.image.fill((255, 0, 0))
        self.start_x = x
        self.start_y = y

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        
        if (0 <= new_x < self.maze.w and 
            0 <= new_y < self.maze.h and 
            not isinstance(self.maze.grid[new_x][new_y], Wall)):
            self.x = new_x
            self.y = new_y
            self.rect.x = self.x * self.w
            self.rect.y = self.y * self.h
            return True
        return False

    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.rect.x = self.x * self.w
        self.rect.y = self.y * self.h

class EndPoint(Cell):
    def __init__(self, x, y, maze):
        super(EndPoint, self).__init__(x, y, maze)
        self.image.fill((0, 255, 0))

class Maze:
    def __init__(self, level):
        # Initialize maze properties
        base_size = 11
        size = min(base_size + (level * 4), 41)
        self.w = size
        self.h = size
        self.grid = None
        self.player = None
        self.end_point = None
        self.level = level
        self.screen = None
        
        # Initialize RL properties
        self.observation_space = np.zeros(4)  # [player_x, player_y, end_x, end_y]
        self.action_space = np.arange(4)  # [up, down, left, right]
        
        # Generate initial maze
        self.reset()
        
    def reset(self):
        """Reset the maze to initial state. Required for RL training."""
        # Initialize grid with walls
        self.grid = [[Wall(x, y, self) for y in range(self.h)] for x in range(self.w)]
        
        # Generate maze layout
        self.generate(animate=False)
        
        # Return initial state
        return self.get_state()
        
    def get_state(self):
        """Get current state representation for RL."""
        return np.array([
            self.player.x / self.w,  # Normalized x position
            self.player.y / self.h,  # Normalized y position
            self.end_point.x / self.w,  # Normalized target x
            self.end_point.y / self.h,  # Normalized target y
        ])
        
    def step(self, action):
        """Execute action and return next state, reward, done. Required for RL training."""
        # Convert action (0,1,2,3) to movement
        action_map = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0)    # right
        }
        dx, dy = action_map[action]
        
        # Execute move
        move_success = self.player.move(dx, dy)
        
        # Calculate reward
        reward = self._get_reward(move_success)
        
        # Check if done
        done = self.check_win()
        
        # Get new state
        next_state = self.get_state()
        
        return next_state, reward, done, {}
    
    def _get_reward(self, move_success):
        """Calculate reward for current state"""
        if self.check_win():
            return 100.0  # Win reward
        if not move_success:
            return -1.0  # Wall collision penalty
        
        # Distance-based reward
        dx = self.player.x - self.end_point.x
        dy = self.player.y - self.end_point.y
        distance = np.sqrt(dx*dx + dy*dy)
        return -0.1 * distance  # Small negative reward based on distance
        
    def get(self, x, y):
        return self.grid[x][y]
    
    def place_wall(self, x, y):
        self.grid[x][y] = Wall(x, y, self)
        
    def draw(self, screen):
        self.screen = screen
        # Clear screen with black
        screen.fill((0, 0, 0))
        
        # Calculate offset to center the maze
        offset_x = (screen.get_width() - (self.w * Cell.w)) // 2
        offset_y = (screen.get_height() - (self.h * Cell.h)) // 2
        
        # Draw grid with offset
        for row in self.grid:
            for cell in row:
                cell.rect.x = offset_x + (cell.x * Cell.w)
                cell.rect.y = offset_y + (cell.y * Cell.h)
                cell.draw(screen)
        
        # Draw end point and player last
        if self.end_point:
            self.end_point.rect.x = offset_x + (self.end_point.x * Cell.w)
            self.end_point.rect.y = offset_y + (self.end_point.y * Cell.h)
            self.end_point.draw(screen)
        if self.player:
            self.player.rect.x = offset_x + (self.player.x * Cell.w)
            self.player.rect.y = offset_y + (self.player.y * Cell.h)
            self.player.draw(screen)
                
    def generate(self, screen=None, animate=False):
        unvisited = [c for r in self.grid for c in r if c.x % 2 and c.y % 2]
        cur = unvisited.pop()
        stack = []
        
        while unvisited:
            try:
                n = choice([c for c in map(lambda x: self.get(*x), cur.nbs) 
                          if c in unvisited])
                stack.append(cur)
                
                nx, ny = cur.x - (cur.x - n.x) // 2, cur.y - (cur.y - n.y) // 2
                self.grid[nx][ny] = Cell(nx, ny, self)
                self.grid[cur.x][cur.y] = Cell(cur.x, cur.y, self)
                
                cur = n
                unvisited.remove(n)
                
                if animate and screen:
                    self.draw(screen)
                    pygame.display.update()
                    pygame.time.wait(10)
                    
            except IndexError:
                if stack:
                    cur = stack.pop()
        
        # Place start and end points
        start_x, start_y = 1, 1
        self.player = Player(start_x, start_y, self)
        self.grid[start_x][start_y] = Cell(start_x, start_y, self)
        
        end_x, end_y = self.w - 2, self.h - 2
        self.end_point = EndPoint(end_x, end_y, self)
        self.grid[end_x][end_y] = Cell(end_x, end_y, self)
        
    def check_win(self):
        return (self.player.x == self.end_point.x and 
                self.player.y == self.end_point.y)


def draw_maze(screen, level):
    maze = Maze(level)
    maze.generate(screen, True)
    return maze

def main():
    pygame.init()
    
    # Fixed window size
    WINSIZE = (Cell.w * 41, Cell.h * 41)  # Maximum maze size
    scr_inf = pygame.display.Info()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '{}, {}'.format(
        scr_inf.current_w // 2 - WINSIZE[0] // 2,
        scr_inf.current_h // 2 - WINSIZE[1] // 2
    )
    
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Progressive Maze Game')
    
    clock = pygame.time.Clock()
    level = 0
    maze = draw_maze(screen, level)
    
    # Add font for level display
    font = pygame.font.Font(None, 36)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                running = False
            elif event.type == KEYUP and event.key == K_SPACE:
                # Reset current level
                maze = draw_maze(screen, level)
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    maze.player.move(-1, 0)
                elif event.key == K_RIGHT:
                    maze.player.move(1, 0)
                elif event.key == K_UP:
                    maze.player.move(0, -1)
                elif event.key == K_DOWN:
                    maze.player.move(0, 1)
                elif event.key == K_r:
                    maze.player.reset()
        
        # Check for win condition
        if maze.check_win():
            level += 1
            maze = draw_maze(screen, level)
        
        # Draw everything
        maze.draw(screen)
        
        # Draw level text
        level_text = font.render(f'Level {level + 1}', True, (255, 255, 255))
        screen.blit(level_text, (10, 10))
        
        pygame.display.update()
        clock.tick(60)
    
    pygame.quit()

if __name__ == '__main__':
    main()