import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import numpy as np
import os
from typing import Optional

# Try to import RL dependencies
try:
    from stable_baselines3 import PPO, DQN, A2C
    RL_AVAILABLE = True
except ImportError:
    print("Stable Baselines3 not available. AI opponent will use rule-based behavior.")
    RL_AVAILABLE = False

class AIOpponent:
    """AI opponent that can use trained RL models or fallback to rule-based behavior"""
    
    def __init__(self, model_path: Optional[str] = None, difficulty: str = 'medium'):
        self.model = None
        self.model_loaded = False
        self.difficulty = difficulty  # 'easy', 'medium', 'hard'
        
        # Load RL model if available
        if model_path and RL_AVAILABLE and os.path.exists(model_path + '.zip'):
            try:
                # Detect model type from filename
                if 'ppo' in model_path.lower():
                    self.model = PPO.load(model_path)
                elif 'dqn' in model_path.lower():
                    self.model = DQN.load(model_path)
                elif 'a2c' in model_path.lower():
                    self.model = A2C.load(model_path)
                
                if self.model:
                    self.model_loaded = True
                    print(f"✅ Loaded AI model: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load AI model: {e}")
                print("Falling back to rule-based AI")
        
        # Difficulty settings for rule-based AI
        self.difficulty_settings = {
            'easy': {'reaction_speed': 0.7, 'error_rate': 0.3, 'prediction': 0.1},
            'medium': {'reaction_speed': 0.85, 'error_rate': 0.15, 'prediction': 0.3},
            'hard': {'reaction_speed': 0.95, 'error_rate': 0.05, 'prediction': 0.6}
        }
    
    def get_action(self, game_state: dict) -> int:
        """Get action from AI opponent (0=stay, 1=up, 2=down)"""
        
        if self.model_loaded:
            return self._get_rl_action(game_state)
        else:
            return self._get_rule_based_action(game_state)
    
    def _get_rl_action(self, game_state: dict) -> int:
        """Get action from trained RL model"""
        try:
            # Convert game state to observation format expected by RL model
            obs = self._game_state_to_observation(game_state)
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        except Exception as e:
            print(f"Error in RL prediction: {e}")
            return self._get_rule_based_action(game_state)
    
    def _get_rule_based_action(self, game_state: dict) -> int:
        """Fallback rule-based AI with difficulty settings"""
        settings = self.difficulty_settings[self.difficulty]
        
        # Extract relevant state
        ball_x = game_state['ball_x']
        ball_y = game_state['ball_y']
        ball_vx = game_state['ball_vx']
        ball_vy = game_state['ball_vy']
        paddle_y = game_state['right_paddle_y']
        paddle_height = game_state['paddle_height']
        
        # Add random errors based on difficulty
        if random.random() < settings['error_rate']:
            return random.choice([0, 1, 2])
        
        # Only react if ball is moving towards AI paddle
        if ball_vx <= 0:
            return 0  # Stay put if ball moving away
        
        # Predict where ball will be when it reaches paddle
        predicted_y = ball_y
        if settings['prediction'] > random.random() and ball_vx > 0:
            time_to_paddle = (0.94 - ball_x) / ball_vx if ball_vx > 0 else 0
            predicted_y = ball_y + ball_vy * time_to_paddle
            
            # Handle ball bouncing off walls
            while predicted_y < 0 or predicted_y > 1:
                if predicted_y < 0:
                    predicted_y = -predicted_y
                elif predicted_y > 1:
                    predicted_y = 2 - predicted_y
        
        # Calculate target paddle position
        target_y = predicted_y
        current_paddle_center = paddle_y + paddle_height / 2
        
        # Apply reaction speed
        if random.random() > settings['reaction_speed']:
            return 0  # Miss some reactions
        
        # Decide action based on target position
        diff = target_y - current_paddle_center
        threshold = 0.05  # Dead zone
        
        if diff > threshold:
            return 1  # Move up
        elif diff < -threshold:
            return 2  # Move down
        else:
            return 0  # Stay
    
    def _game_state_to_observation(self, game_state: dict) -> np.ndarray:
        """Convert game state to RL model observation format"""
        # This should match the observation format used in training
        ball_x = game_state['ball_x']
        ball_y = game_state['ball_y']
        ball_vx = game_state['ball_vx']
        ball_vy = game_state['ball_vy']
        left_paddle_y = game_state['left_paddle_y']
        right_paddle_y = game_state['right_paddle_y']
        left_score = game_state['left_score']
        right_score = game_state['right_score']
        
        # Calculate derived features (matching training environment)
        ball_to_left_paddle_dist = np.sqrt((ball_x - 0.06)**2 + (ball_y - left_paddle_y)**2)
        ball_to_right_paddle_dist = np.sqrt((ball_x - 0.94)**2 + (ball_y - right_paddle_y)**2)
        ball_rel_to_left = ball_y - left_paddle_y
        ball_rel_to_right = ball_y - right_paddle_y
        total_score = left_score + right_score
        score_diff = (left_score - right_score) / 5.0
        
        # Ball direction indicators
        ball_moving_left = 1.0 if ball_vx < 0 else 0.0
        ball_moving_right = 1.0 if ball_vx > 0 else 0.0
        ball_moving_up = 1.0 if ball_vy > 0 else 0.0
        ball_moving_down = 1.0 if ball_vy < 0 else 0.0
        
        # Time to walls
        time_to_left_wall = abs(ball_x / ball_vx) if ball_vx != 0 else 1.0
        time_to_right_wall = abs((1.0 - ball_x) / ball_vx) if ball_vx != 0 else 1.0
        time_to_left_wall = min(time_to_left_wall, 1.0)
        time_to_right_wall = min(time_to_right_wall, 1.0)
        
        # Construct observation (25 features to match training)
        obs = np.array([
            left_paddle_y, right_paddle_y, ball_x, ball_y,
            ball_vx, ball_vy,
            ball_to_left_paddle_dist, ball_to_right_paddle_dist,
            ball_rel_to_left, ball_rel_to_right,
            left_score / 5.0, right_score / 5.0, score_diff, total_score / 10.0,
            ball_moving_left, ball_moving_right, ball_moving_up, ball_moving_down,
            0.0, 0.0,  # Previous actions (not available in game)
            time_to_left_wall, time_to_right_wall,
            abs(ball_vy), 0.0,  # Game progress (not available)
            -1.0  # Right player indicator
        ], dtype=np.float32)
        
        return obs


class PongGameWithAI(pyglet.window.Window):
    def __init__(self, ai_model_path=None, ai_difficulty='medium'):
        super().__init__(800, 600, caption="Pong vs AI")
        
        # Initialize AI opponent
        self.ai_opponent = AIOpponent(ai_model_path, ai_difficulty)
        
        # Game settings
        self.paddle_speed = 300
        self.ball_speed = 250
        
        # Paddle dimensions
        self.paddle_width = 15
        self.paddle_height = 100
        
        # Ball dimensions
        self.ball_size = 15
        
        # Initialize paddles
        self.left_paddle = shapes.Rectangle(
            30, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height,
            color=(100, 255, 100)  # Green for human player
        )
        
        self.right_paddle = shapes.Rectangle(
            self.width - 30 - self.paddle_width, self.height // 2 - self.paddle_height // 2,
            self.paddle_width, self.paddle_height,
            color=(255, 100, 100) if self.ai_opponent.model_loaded else (255, 255, 100)  # Red for AI, Yellow for rule-based
        )
        
        # Initialize ball
        self.ball = shapes.Circle(
            self.width // 2, self.height // 2, self.ball_size,
            color=(255, 255, 255)
        )
        
        # Ball velocity
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-1, 1) * self.ball_speed
        
        # Scores
        self.left_score = 0
        self.right_score = 0
        
        # Key states
        self.keys_pressed = set()
        
        # AI action timing
        self.ai_action_timer = 0
        self.ai_action_interval = 1/30.0  # AI makes decisions at 30 FPS
        
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
            color=(100, 255, 100, 255)
        )
        
        self.right_score_label = pyglet.text.Label(
            str(self.right_score),
            font_name='Arial',
            font_size=36,
            x=3 * self.width // 4,
            y=self.height - 50,
            anchor_x='center',
            color=(255, 100, 100, 255) if self.ai_opponent.model_loaded else (255, 255, 100, 255)
        )
        
        # AI status label
        ai_type = "AI (RL)" if self.ai_opponent.model_loaded else f"AI (Rule-based {ai_difficulty.title()})"
        self.ai_status_label = pyglet.text.Label(
            f"Human vs {ai_type}",
            font_name='Arial',
            font_size=14,
            x=self.width // 2,
            y=self.height - 30,
            anchor_x='center',
            color=(200, 200, 200, 255)
        )
        
        # Instructions label
        self.instructions = pyglet.text.Label(
            "Human Player: W/S keys | ESC: Quit | R: Reset Game",
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
        
        print(f"Game started with AI difficulty: {ai_difficulty}")
        if self.ai_opponent.model_loaded:
            print("🤖 Using trained RL model for AI opponent")
        else:
            print("🎮 Using rule-based AI opponent")
    
    def get_game_state(self):
        """Get current game state for AI opponent"""
        return {
            'ball_x': self.ball.x / self.width,
            'ball_y': self.ball.y / self.height,
            'ball_vx': self.ball_dx / self.ball_speed,
            'ball_vy': self.ball_dy / self.ball_speed,
            'left_paddle_y': self.left_paddle.y / (self.height - self.paddle_height),
            'right_paddle_y': self.right_paddle.y / (self.height - self.paddle_height),
            'paddle_height': self.paddle_height / self.height,
            'left_score': self.left_score,
            'right_score': self.right_score,
            'width': self.width,
            'height': self.height
        }
    
    def on_key_press(self, symbol, modifiers):
        self.keys_pressed.add(symbol)
        
        if symbol == key.ESCAPE:
            self.close()
        elif symbol == key.R:
            self.reset_game()
    
    def on_key_release(self, symbol, modifiers):
        self.keys_pressed.discard(symbol)
    
    def reset_game(self):
        """Reset the game"""
        self.left_score = 0
        self.right_score = 0
        
        # Reset paddle positions
        self.left_paddle.y = self.height // 2 - self.paddle_height // 2
        self.right_paddle.y = self.height // 2 - self.paddle_height // 2
        
        # Reset ball
        self.reset_ball()
        
        print("Game reset!")
    
    def update(self, dt):
        # Handle human paddle movement (left paddle)
        if key.W in self.keys_pressed:
            self.left_paddle.y = min(self.height - self.paddle_height, 
                                   self.left_paddle.y + self.paddle_speed * dt)
        if key.S in self.keys_pressed:
            self.left_paddle.y = max(0, self.left_paddle.y - self.paddle_speed * dt)
        
        # Handle AI paddle movement (right paddle)
        self.ai_action_timer += dt
        if self.ai_action_timer >= self.ai_action_interval:
            game_state = self.get_game_state()
            ai_action = self.ai_opponent.get_action(game_state)
            
            # Apply AI action
            if ai_action == 1:  # Up
                self.right_paddle.y = min(self.height - self.paddle_height,
                                        self.right_paddle.y + self.paddle_speed * dt * 3)
            elif ai_action == 2:  # Down
                self.right_paddle.y = max(0, self.right_paddle.y - self.paddle_speed * dt * 3)
            # ai_action == 0 means stay (no movement)
            
            self.ai_action_timer = 0
        
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
        if self.ball.x < 0:
            self.right_score += 1
            print(f"AI scores! Score: Human {self.left_score} - {self.right_score} AI")
            self.reset_ball()
        elif self.ball.x > self.width:
            self.left_score += 1
            print(f"Human scores! Score: Human {self.left_score} - {self.right_score} AI")
            self.reset_ball()
        
        # Check for game over
        if self.left_score >= 10:
            print("🎉 Human wins the match!")
            self.reset_game()
        elif self.right_score >= 10:
            print("🤖 AI wins the match!")
            self.reset_game()
        
        # Update score labels
        self.left_score_label.text = str(self.left_score)
        self.right_score_label.text = str(self.right_score)
    
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
    
    def reset_ball(self):
        # Reset ball to center
        self.ball.x = self.width // 2
        self.ball.y = self.height // 2
        
        # Random direction for ball
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-0.5, 0.5) * self.ball_speed
    
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
        
        # Draw AI status
        self.ai_status_label.draw()
        
        # Draw instructions
        self.instructions.draw()


def main():
    """Main function to start the game"""
    print("Pong Game with AI Opponent")
    print("=" * 40)
    
    # Check for available AI models
    models_dir = "./pong_models"
    available_models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('_final.zip'):
                model_path = os.path.join(models_dir, file[:-4])  # Remove .zip extension
                available_models.append(model_path)
    
    # Game configuration
    ai_model_path = None
    ai_difficulty = 'medium'
    
    if available_models:
        print("Available AI models:")
        for i, model in enumerate(available_models):
            print(f"  {i+1}. {os.path.basename(model)}")
        print(f"  {len(available_models)+1}. Use rule-based AI")
        
        try:
            choice = input(f"Select AI opponent (1-{len(available_models)+1}, default: rule-based): ").strip()
            if choice and choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    ai_model_path = available_models[choice_idx]
        except:
            pass
    
    if not ai_model_path:
        print("Available difficulties: easy, medium, hard")
        difficulty_input = input("Select difficulty (default: medium): ").strip().lower()
        if difficulty_input in ['easy', 'medium', 'hard']:
            ai_difficulty = difficulty_input
    
    # Start the game
    print(f"\nStarting game...")
    if ai_model_path:
        print(f"AI Model: {os.path.basename(ai_model_path)}")
    else:
        print(f"Rule-based AI: {ai_difficulty}")
    
    print("\nControls:")
    print("  W/S - Move your paddle (left side)")
    print("  R - Reset game")
    print("  ESC - Quit")
    print("\nFirst to 10 points wins!")
    
    try:
        game = PongGameWithAI(ai_model_path, ai_difficulty)
        pyglet.app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install pyglet stable-baselines3[extra] numpy")


if __name__ == "__main__":
    main()