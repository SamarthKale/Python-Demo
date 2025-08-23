import gymnasium as gym # CHANGE 1: Import gymnasium instead of gym
from gymnasium import spaces # CHANGE 2: Import spaces from gymnasium
import numpy as np
import random

# CHANGE 3: The class now inherits from gym.Env (which is now gymnasium.Env)
class PongEnv(gym.Env):
    """Custom Environment for Pong that follows the gymnasium interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PongEnv, self).__init__()

        # Game constants
        self.width = 800
        self.height = 600
        self.paddle_height = 100
        self.paddle_speed = 300
        self.ball_speed = 250
        self.max_score = 5

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        # Initialize state (no change needed here)
        # The first call to reset() will properly set up the state.

    def _get_obs(self):
        """Constructs the observation array from the current game state."""
        # This function remains the same as it correctly calculates the state vector.
        ball_x_norm = self.ball_x / self.width
        ball_y_norm = self.ball_y / self.height
        left_paddle_y_norm = self.left_paddle_y / (self.height - self.paddle_height)
        right_paddle_y_norm = self.right_paddle_y / (self.height - self.paddle_height)
        ball_vx_norm = self.ball_dx / self.ball_speed
        ball_vy_norm = self.ball_dy / self.ball_speed

        ball_to_left_paddle_dist = np.sqrt((ball_x_norm - 0.06)**2 + (ball_y_norm - left_paddle_y_norm)**2)
        ball_to_right_paddle_dist = np.sqrt((ball_x_norm - 0.94)**2 + (ball_y_norm - right_paddle_y_norm)**2)
        ball_rel_to_left = ball_y_norm - left_paddle_y_norm
        ball_rel_to_right = ball_y_norm - right_paddle_y_norm
        score_diff = (self.right_score - self.left_score) / self.max_score
        total_score = (self.left_score + self.right_score) / (2 * self.max_score)

        ball_moving_left = 1.0 if self.ball_dx < 0 else 0.0
        ball_moving_right = 1.0 if self.ball_dx > 0 else 0.0
        ball_moving_up = 1.0 if self.ball_dy > 0 else 0.0
        ball_moving_down = 1.0 if self.ball_dy < 0 else 0.0

        time_to_left_wall = min(abs(self.ball_x / self.ball_dx), 1.0) if self.ball_dx != 0 else 1.0
        time_to_right_wall = min(abs((self.width - self.ball_x) / self.ball_dx), 1.0) if self.ball_dx != 0 else 1.0

        obs = np.array([
            left_paddle_y_norm, right_paddle_y_norm, ball_x_norm, ball_y_norm,
            ball_vx_norm, ball_vy_norm,
            ball_to_left_paddle_dist, ball_to_right_paddle_dist,
            ball_rel_to_left, ball_rel_to_right,
            self.left_score / self.max_score, self.right_score / self.max_score,
            score_diff, total_score,
            ball_moving_left, ball_moving_right, ball_moving_up, ball_moving_down,
            0.0, 0.0,
            time_to_left_wall, time_to_right_wall,
            abs(ball_vy_norm), 0.0,
            -1.0
        ], dtype=np.float32)
        return obs

    def _rule_based_opponent_action(self):
        """A simple rule-based opponent for the left paddle."""
        if self.left_paddle_y + self.paddle_height / 2 < self.ball_y - 10:
            return 1
        elif self.left_paddle_y + self.paddle_height / 2 > self.ball_y + 10:
            return 2
        return 0

    def step(self, action):
        dt = 1 / 60.0
        reward = 0
        
        # CHANGE 4: Gymnasium `step` returns 5 values. `terminated` is for terminal states
        # (like winning/losing). `truncated` is for time limits.
        terminated = False
        truncated = False # We don't have a time limit, so this is always False.
        info = {}

        # Update paddles
        if action == 1:
            self.right_paddle_y = min(self.height - self.paddle_height, self.right_paddle_y + self.paddle_speed * dt)
        elif action == 2:
            self.right_paddle_y = max(0, self.right_paddle_y - self.paddle_speed * dt)

        opponent_action = self._rule_based_opponent_action()
        if opponent_action == 1:
            self.left_paddle_y = min(self.height - self.paddle_height, self.left_paddle_y + self.paddle_speed * dt)
        elif opponent_action == 2:
            self.left_paddle_y = max(0, self.left_paddle_y - self.paddle_speed * dt)

        # Update ball and collisions
        self.ball_x += self.ball_dx * dt
        self.ball_y += self.ball_dy * dt

        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_dy *= -1

        if self.ball_dx < 0 and 30 <= self.ball_x <= 45 and self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height:
            self.ball_dx *= -1.1
        if self.ball_dx > 0 and self.width - 45 <= self.ball_x <= self.width - 30 and self.right_paddle_y <= self.ball_y <= self.right_paddle_y + self.paddle_height:
            self.ball_dx *= -1.1
            reward += 1.0

        # Scoring
        if self.ball_x < 0:
            self.right_score += 1
            reward += 10.0
            self.reset_ball()
        elif self.ball_x > self.width:
            self.left_score += 1
            reward -= 10.0
            self.reset_ball()
            
        reward += 0.1 * (1 - abs(self.right_paddle_y + self.paddle_height/2 - self.ball_y) / self.height)

        # Check for game end
        if self.left_score >= self.max_score or self.right_score >= self.max_score:
            terminated = True

        # CHANGE 5: The return signature must match the new API.
        return self._get_obs(), reward, terminated, truncated, info

    def reset_ball(self):
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-1, 1) * self.ball_speed
        self.ball_dx = np.clip(self.ball_dx, -self.ball_speed*2, self.ball_speed*2)

    def reset(self, seed=None, options=None):
        # CHANGE 6: It's good practice to handle the seed and call the superclass's reset.
        super().reset(seed=seed)
        
        self.left_paddle_y = self.height / 2 - self.paddle_height / 2
        self.right_paddle_y = self.height / 2 - self.paddle_height / 2
        self.left_score = 0
        self.right_score = 0
        self.reset_ball()
        
        # CHANGE 7: THIS IS THE FIX FOR YOUR ERROR. `reset` must return obs and an info dict.
        return self._get_obs(), {}

    def render(self):
        # Rendering is handled externally, so this can be left empty.
        pass

    def close(self):
        pass
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Configuration ---
# Training parameters
TOTAL_TIMESTEPS = 1_000_000  # Increase for better performance (e.g., 1,000,000)
MODEL_ALGO = "PPO"
MODEL_NAME = "ppo_pong_model_v1"

# Directories
models_dir = "./pong_models"
log_dir = "./pong_logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Path to the final model file
save_path = os.path.join(models_dir, f"{MODEL_NAME}_final")

def train_agent():
    """
    Trains the RL agent and saves the model.
    """
    print("Initializing Pong environment...")
    # Create and vectorize the environment
    env = DummyVecEnv([lambda: PongEnv()])

    print(f"Using algorithm: {MODEL_ALGO}")
    # Initialize the PPO model
    # 'MlpPolicy' is a standard feedforward neural network.
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01
    )

    # Set up a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=models_dir,
        name_prefix=MODEL_NAME
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"Models will be saved in: {models_dir}")
    print(f"Logs will be saved in: {log_dir}")

    # --- Start Training ---
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name=MODEL_NAME
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")

    # --- Save the Final Model ---
    print(f"Training finished. Saving final model to {save_path}.zip")
    model.save(save_path)
    print("✅ Model saved successfully!")

if __name__ == '__main__':
    # Check for necessary libraries
    try:
        import gym
        import stable_baselines3
    except ImportError:
        print("❌ Error: Required libraries not found.")
        print("Please install them by running:")
        print("pip install stable-baselines3[extra] gym")
    else:
        train_agent()