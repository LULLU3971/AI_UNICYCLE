import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# --- Import the Custom Environment ---
# Ensure the file containing your UnicycleEnv class is named 'environment.py'
# and is in the same directory as this training script.
from environment import UnicycleEnv

# --- Configuration ---
# Directories for logs and saved models
TENSORBOARD_LOG_DIR = "tensorboard_logs/"
CHECKPOINT_DIR = "checkpoints/"
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training parameters
TOTAL_TIMESTEPS = 4_000_000
CHECKPOINT_SAVE_FREQ = 10_000 # Save a checkpoint every 10,000 steps

# --- Environment Setup ---
# Using make_vec_env is good practice, especially for more complex environments
# or when scaling up to use multiple environments in parallel (n_envs > 1).
env = make_vec_env(UnicycleEnv, n_envs=1, env_kwargs={'render_mode': None})

# --- Callback for Periodic Model Saving ---
# This callback saves a checkpoint of the model every `CHECKPOINT_SAVE_FREQ` steps.
# The files are saved in the CHECKPOINT_DIR with a name prefix 'ppo_unicycle'.
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_SAVE_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_unicycle",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# --- Resume from Checkpoint Logic ---
# Find the latest checkpoint file in the directory.
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_unicycle_*.zip"))
latest_checkpoint = max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None

if latest_checkpoint:
    print(f"Resuming training from the latest checkpoint: {latest_checkpoint}")
    # Load the existing model
    model = PPO.load(
        latest_checkpoint,
        env=env,
        tensorboard_log=TENSORBOARD_LOG_DIR
    )
else:
    print("Starting a new training session.")
    # Create a new PPO model if no checkpoint is found
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        n_epochs=10,
        gae_lambda=0.95,
        vf_coef=0.5
    )


# --- Training ---
print("Starting model training...")
# The learn method will run for the total number of timesteps.
# The CheckpointCallback will be triggered periodically during training.
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    tb_log_name="PPO_Unicycle",
    reset_num_timesteps=not bool(latest_checkpoint) # Reset counter if not resuming
)
print("Training finished.")

# --- Save the Final Model ---
FINAL_MODEL_PATH = "final_model.zip"
model.save(FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")

# --- Cleanup ---
env.close()