import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import numpy as np
import json
import csv
import os
from datetime import datetime
import time

class PongDatasetGenerator(pyglet.window.Window):
    def __init__(self, collect_data=True, save_interval=1000):
        super().__init__(800, 600, caption="Pong Dataset Generator")
        
        # Dataset collection settings
        self.collect_data = collect_data
        self.save_interval = save_interval
        self.data_collected = []
        self.episode_data = []
        self.current_episode = 0
        self.step_counter = 0
        self.episode_start_time = time.time()
        
        # Game settings
        self.paddle_speed = 300
        self.ball_speed = 250
        self.max_score = 5  # Episode ends when someone reaches this score
        
        # Paddle dimensions
        self.paddle_width = 15
        self.paddle_height = 100
        
        # Ball dimensions
        self.ball_size = 15
        
        # Initialize paddles
        self.left_paddle = shapes.Rectangle(
            30, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height,
            color=(255, 255, 255)
        )
        
        self.right_paddle = shapes.Rectangle(
            self.width - 30 - self.paddle_width, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height,
            color=(255, 255, 255)
        )
        
        # Initialize ball
        self.ball = shapes.Circle(
            self.width // 2, self.height // 2, self.ball_size,
            color=(255, 255, 255)
        )
        
        # Ball velocity
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-1, 1) * self.ball_speed
        
        # Previous ball position for velocity calculation
        self.prev_ball_x = self.ball.x
        self.prev_ball_y = self.ball.y
        
        # Scores
        self.left_score = 0
        self.right_score = 0
        
        # Key states
        self.keys_pressed = set()
        
        # Previous actions for temporal information
        self.prev_left_action = 0  # 0: no action, 1: up, -1: down
        self.prev_right_action = 0
        
        # Game state tracking
        self.game_over = False
        self.winner = None
        
        # Create center line
        self.center_line = []
        for i in range(0, self.height, 20):
            self.center_line.append(
                shapes.Rectangle(self.width // 2 - 2, i, 4, 10, color=(255, 255, 255))
            )
        
        # Labels for scores
        self.left_score_label = pyglet.text.Label(
            str(self.left_score),
            font_name='Arial',
            font_size=36,
            x=self.width // 4,
            y=self.height - 50,
            anchor_x='center',
            color=(255, 255, 255, 255)
        )
        
        self.right_score_label = pyglet.text.Label(
            str(self.right_score),
            font_name='Arial',
            font_size=36,
            x=3 * self.width // 4,
            y=self.height - 50,
            anchor_x='center',
            color=(255, 255, 255, 255)
        )
        
        # Dataset info label
        self.dataset_info = pyglet.text.Label(
            f"Episode: {self.current_episode} | Steps: {self.step_counter} | Data Points: {len(self.data_collected)}",
            font_name='Arial',
            font_size=12,
            x=10,
            y=self.height - 30,
            anchor_x='left',
            color=(200, 255, 200, 255)
        )
        
        # Instructions label
        self.instructions = pyglet.text.Label(
            "Left: W/S | Right: UP/DOWN | R: Reset | ESC: Save & Quit",
            font_name='Arial',
            font_size=12,
            x=self.width // 2,
            y=30,
            anchor_x='center',
            color=(200, 200, 200, 255)
        )
        
        # Set background color to black
        pyglet.gl.glClearColor(0, 0, 0, 1)
        
        # Schedule update
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        
        # Create dataset directory
        self.dataset_dir = "pong_dataset"
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        print("Pong Dataset Generator Started!")
        print(f"Data will be saved to: {self.dataset_dir}/")
        print("Play the game normally to generate training data.")
    
    def get_game_state(self):
        """Extract comprehensive game state for RL training"""
        # Normalize positions to [0, 1] range
        left_paddle_y_norm = self.left_paddle.y / (self.height - self.paddle_height)
        right_paddle_y_norm = self.right_paddle.y / (self.height - self.paddle_height)
        ball_x_norm = self.ball.x / self.width
        ball_y_norm = self.ball.y / self.height
        
        # Ball velocity (normalized)
        ball_vx_norm = self.ball_dx / self.ball_speed
        ball_vy_norm = self.ball_dy / self.ball_speed
        
        # Distance calculations (useful for RL)
        ball_to_left_paddle_dist = np.sqrt(
            (self.ball.x - (self.left_paddle.x + self.paddle_width/2))**2 + 
            (self.ball.y - (self.left_paddle.y + self.paddle_height/2))**2
        ) / np.sqrt(self.width**2 + self.height**2)  # Normalize
        
        ball_to_right_paddle_dist = np.sqrt(
            (self.ball.x - (self.right_paddle.x + self.paddle_width/2))**2 + 
            (self.ball.y - (self.right_paddle.y + self.paddle_height/2))**2
        ) / np.sqrt(self.width**2 + self.height**2)  # Normalize
        
        # Time to impact calculations (simplified)
        time_to_left_wall = abs(self.ball.x / self.ball_dx) if self.ball_dx != 0 else float('inf')
        time_to_right_wall = abs((self.width - self.ball.x) / self.ball_dx) if self.ball_dx != 0 else float('inf')
        
        # Relative positions (ball relative to paddles)
        ball_rel_to_left = (self.ball.y - (self.left_paddle.y + self.paddle_height/2)) / self.height
        ball_rel_to_right = (self.ball.y - (self.right_paddle.y + self.paddle_height/2)) / self.height
        
        # Game progress indicators
        score_diff = (self.left_score - self.right_score) / self.max_score
        total_score = (self.left_score + self.right_score) / (2 * self.max_score)
        
        # Ball direction indicators
        ball_moving_left = 1 if self.ball_dx < 0 else 0
        ball_moving_right = 1 if self.ball_dx > 0 else 0
        ball_moving_up = 1 if self.ball_dy > 0 else 0
        ball_moving_down = 1 if self.ball_dy < 0 else 0
        
        state = {
            # Basic positions (normalized)
            'left_paddle_y': left_paddle_y_norm,
            'right_paddle_y': right_paddle_y_norm,
            'ball_x': ball_x_norm,
            'ball_y': ball_y_norm,
            
            # Ball velocity
            'ball_vx': ball_vx_norm,
            'ball_vy': ball_vy_norm,
            
            # Distances
            'ball_to_left_paddle_dist': ball_to_left_paddle_dist,
            'ball_to_right_paddle_dist': ball_to_right_paddle_dist,
            
            # Relative positions
            'ball_rel_to_left_paddle': ball_rel_to_left,
            'ball_rel_to_right_paddle': ball_rel_to_right,
            
            # Game state
            'left_score': self.left_score,
            'right_score': self.right_score,
            'score_difference': score_diff,
            'game_progress': total_score,
            
            # Ball direction
            'ball_moving_left': ball_moving_left,
            'ball_moving_right': ball_moving_right,
            'ball_moving_up': ball_moving_up,
            'ball_moving_down': ball_moving_down,
            
            # Previous actions (for temporal context)
            'prev_left_action': self.prev_left_action,
            'prev_right_action': self.prev_right_action,
            
            # Time information
            'time_to_left_wall': min(time_to_left_wall, 10.0),  # Cap at 10 seconds
            'time_to_right_wall': min(time_to_right_wall, 10.0),
        }
        
        return state
    
    def calculate_rewards(self, prev_state, current_state, left_action, right_action):
        """Calculate rewards for both players based on state transition"""
        left_reward = 0
        right_reward = 0
        
        # Scoring rewards
        if self.left_score > prev_state.get('left_score', 0):
            left_reward += 10
            right_reward -= 10
        elif self.right_score > prev_state.get('right_score', 0):
            left_reward -= 10
            right_reward += 10
        
        # Paddle positioning rewards (encourage good defensive positioning)
        ball_y = current_state['ball_y'] * self.height
        left_paddle_center = current_state['left_paddle_y'] * (self.height - self.paddle_height) + self.paddle_height/2
        right_paddle_center = current_state['right_paddle_y'] * (self.height - self.paddle_height) + self.paddle_height/2
        
        # Reward for keeping paddle aligned with ball
        left_alignment = 1 - abs(ball_y - left_paddle_center) / (self.height/2)
        right_alignment = 1 - abs(ball_y - right_paddle_center) / (self.height/2)
        
        # Only reward alignment when ball is approaching
        if current_state['ball_moving_left']:
            left_reward += left_alignment * 0.1
        if current_state['ball_moving_right']:
            right_reward += right_alignment * 0.1
        
        # Penalty for unnecessary movement (energy efficiency)
        if left_action != 0:
            left_reward -= 0.01
        if right_action != 0:
            right_reward -= 0.01
        
        # Bonus for successful paddle hits (collision detection would be ideal)
        # This is approximated by checking if ball changed direction near paddle
        if hasattr(self, 'ball_hit_left') and self.ball_hit_left:
            left_reward += 1
            self.ball_hit_left = False
            
        if hasattr(self, 'ball_hit_right') and self.ball_hit_right:
            right_reward += 1
            self.ball_hit_right = False
        
        return left_reward, right_reward
    
    def collect_data_point(self, state, left_action, right_action, next_state, left_reward, right_reward, done):
        """Collect a single data point for the dataset"""
        data_point = {
            'episode': self.current_episode,
            'step': self.step_counter,
            'timestamp': time.time(),
            
            # State information
            'state': state,
            'next_state': next_state,
            
            # Actions
            'left_action': left_action,
            'right_action': right_action,
            
            # Rewards
            'left_reward': left_reward,
            'right_reward': right_reward,
            
            # Episode information
            'done': done,
            'winner': self.winner if done else None,
        }
        
        self.episode_data.append(data_point)
        
        if done:
            # Add episode-level information
            episode_summary = {
                'episode': self.current_episode,
                'total_steps': len(self.episode_data),
                'final_score_left': self.left_score,
                'final_score_right': self.right_score,
                'winner': self.winner,
                'episode_duration': time.time() - self.episode_start_time,
                'data_points': self.episode_data.copy()
            }
            
            self.data_collected.append(episode_summary)
            self.episode_data = []
            
            # Auto-save periodically
            if len(self.data_collected) % self.save_interval == 0:
                self.save_dataset()
    
    def on_key_press(self, symbol, modifiers):
        self.keys_pressed.add(symbol)
        
        if symbol == key.ESCAPE:
            self.save_dataset()
            self.close()
        elif symbol == key.R:
            self.reset_game()
    
    def on_key_release(self, symbol, modifiers):
        self.keys_pressed.discard(symbol)
    
    def reset_game(self):
        """Reset game for new episode"""
        if not self.game_over:
            self.current_episode += 1
        
        self.left_score = 0
        self.right_score = 0
        self.game_over = False
        self.winner = None
        self.step_counter = 0
        self.episode_start_time = time.time()
        
        # Reset paddle positions
        self.left_paddle.y = self.height // 2 - self.paddle_height // 2
        self.right_paddle.y = self.height // 2 - self.paddle_height // 2
        
        # Reset ball
        self.reset_ball()
        
        print(f"Starting Episode {self.current_episode}")
    
    def update(self, dt):
        if self.game_over:
            return
        
        # Store previous state
        prev_state = self.get_game_state()
        
        # Determine actions
        left_action = 0
        right_action = 0
        
        # Handle paddle movement and record actions
        if key.W in self.keys_pressed:
            self.left_paddle.y = min(self.height - self.paddle_height, 
                                   self.left_paddle.y + self.paddle_speed * dt)
            left_action = 1
        elif key.S in self.keys_pressed:
            self.left_paddle.y = max(0, self.left_paddle.y - self.paddle_speed * dt)
            left_action = -1
        
        if key.UP in self.keys_pressed:
            self.right_paddle.y = min(self.height - self.paddle_height,
                                    self.right_paddle.y + self.paddle_speed * dt)
            right_action = 1
        elif key.DOWN in self.keys_pressed:
            self.right_paddle.y = max(0, self.right_paddle.y - self.paddle_speed * dt)
            right_action = -1
        
        # Update ball position
        self.ball.x += self.ball_dx * dt
        self.ball.y += self.ball_dy * dt
        
        # Ball collision with top and bottom walls
        if self.ball.y <= self.ball_size or self.ball.y >= self.height - self.ball_size:
            self.ball_dy = -self.ball_dy
            self.ball.y = max(self.ball_size, min(self.height - self.ball_size, self.ball.y))
        
        # Ball collision with paddles
        self.check_paddle_collision()
        
        # Ball out of bounds (scoring)
        scored = False
        if self.ball.x < 0:
            self.right_score += 1
            scored = True
            self.reset_ball()
        elif self.ball.x > self.width:
            self.left_score += 1
            scored = True
            self.reset_ball()
        
        # Check for game over
        if self.left_score >= self.max_score:
            self.game_over = True
            self.winner = 'left'
        elif self.right_score >= self.max_score:
            self.game_over = True
            self.winner = 'right'
        
        # Get current state after updates
        current_state = self.get_game_state()
        
        # Calculate rewards
        left_reward, right_reward = self.calculate_rewards(prev_state, current_state, left_action, right_action)
        
        # Collect data point
        if self.collect_data:
            self.collect_data_point(
                prev_state, left_action, right_action, current_state,
                left_reward, right_reward, self.game_over
            )
        
        # Update step counter and previous actions
        self.step_counter += 1
        self.prev_left_action = left_action
        self.prev_right_action = right_action
        
        # Update score labels
        self.left_score_label.text = str(self.left_score)
        self.right_score_label.text = str(self.right_score)
        
        # Update dataset info
        self.dataset_info.text = f"Episode: {self.current_episode} | Steps: {self.step_counter} | Episodes Collected: {len(self.data_collected)}"
        
        # Auto-reset if game is over
        if self.game_over:
            print(f"Episode {self.current_episode} completed! Winner: {self.winner}")
            print(f"Final Score - Left: {self.left_score}, Right: {self.right_score}")
            time.sleep(1)  # Brief pause
            self.reset_game()
    
    def check_paddle_collision(self):
        # Left paddle collision
        if (self.ball.x - self.ball_size <= self.left_paddle.x + self.paddle_width and
            self.ball.x + self.ball_size >= self.left_paddle.x and
            self.ball.y + self.ball_size >= self.left_paddle.y and
            self.ball.y - self.ball_size <= self.left_paddle.y + self.paddle_height):
            
            if self.ball_dx < 0:  # Only reverse if ball is moving toward paddle
                self.ball_dx = -self.ball_dx
                # Add some spin based on where the ball hits the paddle
                hit_pos = (self.ball.y - (self.left_paddle.y + self.paddle_height/2)) / (self.paddle_height/2)
                self.ball_dy += hit_pos * 100
                # Ensure ball is outside paddle
                self.ball.x = self.left_paddle.x + self.paddle_width + self.ball_size
                self.ball_hit_left = True
        
        # Right paddle collision
        if (self.ball.x + self.ball_size >= self.right_paddle.x and
            self.ball.x - self.ball_size <= self.right_paddle.x + self.paddle_width and
            self.ball.y + self.ball_size >= self.right_paddle.y and
            self.ball.y - self.ball_size <= self.right_paddle.y + self.paddle_height):
            
            if self.ball_dx > 0:  # Only reverse if ball is moving toward paddle
                self.ball_dx = -self.ball_dx
                # Add some spin based on where the ball hits the paddle
                hit_pos = (self.ball.y - (self.right_paddle.y + self.paddle_height/2)) / (self.paddle_height/2)
                self.ball_dy += hit_pos * 100
                # Ensure ball is outside paddle
                self.ball.x = self.right_paddle.x - self.ball_size
                self.ball_hit_right = True
    
    def reset_ball(self):
        # Reset ball to center
        self.ball.x = self.width // 2
        self.ball.y = self.height // 2
        
        # Random direction for ball
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-0.5, 0.5) * self.ball_speed
    
    def save_dataset(self):
        """Save collected dataset to files"""
        if not self.data_collected:
            print("No data to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON (detailed format)
        json_filename = f"{self.dataset_dir}/pong_dataset_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.data_collected, f, indent=2)
        
        # Save as CSV (flattened format for easy analysis)
        csv_filename = f"{self.dataset_dir}/pong_dataset_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            if self.data_collected:
                # Flatten the data for CSV
                flattened_data = []
                for episode in self.data_collected:
                    for data_point in episode['data_points']:
                        flat_point = {
                            'episode': data_point['episode'],
                            'step': data_point['step'],
                            'left_action': data_point['left_action'],
                            'right_action': data_point['right_action'],
                            'left_reward': data_point['left_reward'],
                            'right_reward': data_point['right_reward'],
                            'done': data_point['done'],
                            'winner': data_point['winner'],
                        }
                        
                        # Add state features
                        for key, value in data_point['state'].items():
                            flat_point[f'state_{key}'] = value
                        
                        # Add next state features
                        for key, value in data_point['next_state'].items():
                            flat_point[f'next_state_{key}'] = value
                        
                        flattened_data.append(flat_point)
                
                if flattened_data:
                    writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_data)
        
        # Save summary statistics
        summary_filename = f"{self.dataset_dir}/dataset_summary_{timestamp}.txt"
        with open(summary_filename, 'w') as f:
            total_episodes = len(self.data_collected)
            total_steps = sum(len(ep['data_points']) for ep in self.data_collected)
            left_wins = sum(1 for ep in self.data_collected if ep['winner'] == 'left')
            right_wins = sum(1 for ep in self.data_collected if ep['winner'] == 'right')
            
            f.write(f"Pong Dataset Summary\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Steps: {total_steps}\n")
            f.write(f"Left Player Wins: {left_wins}\n")
            f.write(f"Right Player Wins: {right_wins}\n")
            f.write(f"Average Steps per Episode: {total_steps/total_episodes if total_episodes > 0 else 0:.2f}\n")
        
        print(f"Dataset saved!")
        print(f"  JSON: {json_filename}")
        print(f"  CSV: {csv_filename}")
        print(f"  Summary: {summary_filename}")
        print(f"  Total Episodes: {len(self.data_collected)}")
        print(f"  Total Data Points: {sum(len(ep['data_points']) for ep in self.data_collected)}")
    
    def on_draw(self):
        self.clear()
        
        # Draw center line
        for line_segment in self.center_line:
            line_segment.draw()
        
        # Draw paddles
        self.left_paddle.draw()
        self.right_paddle.draw()
        
        # Draw ball
        self.ball.draw()
        
        # Draw scores
        self.left_score_label.draw()
        self.right_score_label.draw()
        
        # Draw dataset info
        self.dataset_info.draw()
        
        # Draw instructions
        self.instructions.draw()

if __name__ == "__main__":
    print("Pong RL Dataset Generator")
    print("=" * 50)
    print("This will generate a comprehensive dataset for training RL agents.")
    print("\nDataset includes:")
    print("- Normalized game states (positions, velocities, distances)")
    print("- Player actions and rewards")
    print("- Episode information and outcomes")
    print("- Temporal context (previous actions)")
    print("\nControls:")
    print("- Left Player: W (up) / S (down)")
    print("- Right Player: UP/DOWN arrows") 
    print("- R: Reset current game")
    print("- ESC: Save dataset and quit")
    print("\nPlay multiple games to generate diverse training data!")
    print("Data will be automatically saved every 1000 episodes.")
    
    try:
        game = PongDatasetGenerator(collect_data=True, save_interval=100)
        pyglet.app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed: pip install pyglet numpy")