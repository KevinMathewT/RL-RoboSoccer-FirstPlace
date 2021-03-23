from .train import RoboSoccerGym, MODEL_SAVE_LOCATION
from ..utils import get_score, game_render
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

import warnings
warnings.filterwarnings("ignore")


print("Checking Environment...")
check_env(env=RoboSoccerGym(env_name="Robosoccer-v1", n_players=10), warn=True)
print("Environment Check Passed.\n")

n_players = 10
env = RoboSoccerGym(env_name="Robosoccer-v1", n_players=n_players)
model = A2C.load(MODEL_SAVE_LOCATION)

print(f"\nNumber of Players:               {n_players}")
print("\nRendering Game")
total_reward = game_render(env, model)
print(f"Total Reward for Rendered Game:     {total_reward}")
