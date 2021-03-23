from ..utils import get_score
import random
import numpy as np

import gym
import robo_soccer
import numpy as np
from gym.spaces import Box
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

MODEL_SAVE_LOCATION = "./generated/DeepMinds-Robosoccer10v10-v1"

env = gym.make("Robosoccer-v1")

# Custom RoboSoccerGym for changing the observation space.
class RoboSoccerGym(gym.Env):
    def __init__(self, env_name, n_players, config=None, modify_to_distances=True, modify_to_field_image=False):
        super(RoboSoccerGym, self).__init__()
        self.env = gym.make(env_name)
        self.number_players = n_players
        self.use_field_image = True
        self.add_velocity = False
        self.modify_to_distances = modify_to_distances
        self.modify_to_field_image = modify_to_field_image
        self.observation_size = (
            3 * (1 * self.number_players)) + 3 + (2 * (1 * self.number_players))
        self.observation_space = Box(
            low=-2, high=2, shape=(self.observation_size,))
        self.action_space = env.action_space

    def get_distances(self, raw_obs, number_players):
        assert (2 * number_players + 1) * 4 == len(raw_obs)
        obs = raw_obs.reshape(-1, 4)

        ret = []

        ret.append(abs(1 - obs[0][0]) + abs(obs[0][1] - 0))
        ret.append(1 - obs[0][0])
        ret.append(obs[0][1] - 0)

        # ret.append(abs(obs[0][0] - 1) + abs(obs[0][1] - 0))
        # ret.append(obs[0][0] - 1)
        # ret.append(obs[0][1] - 0)

        for i in range(1, 1 + number_players):
            ret.append(abs(obs[i][0] - obs[0][0]) + abs(obs[i][1] - obs[0][1]))
            ret.append(obs[i][0] - obs[0][0])
            ret.append(obs[i][1] - obs[0][1])

        # for i in range(1 + number_players, 2 * number_players + 1):
        #   ret.append(abs(obs[i][0] - obs[0][0]) + abs(obs[i][1] - obs[0][1]))
        #   ret.append(obs[i][0] - obs[0][0])
        #   ret.append(obs[i][1] - obs[0][1])

        for i in range(1 + number_players, 2 * number_players + 1):
            ret.append(obs[i][0])
            ret.append(obs[i][1])

        ret = np.array(ret)
        # print(ret)
        return ret

    def get_field_image(self, raw_obs, number_players, add_velocity=True):
        assert (2 * number_players + 1) * 4 == len(raw_obs)
        obs = raw_obs.reshape(-1, 4)
        ret = np.zeros((105, 68, 9)) if add_velocity else np.zeros(
            (105, 68, 3))

        for i in range(obs.shape[0]):
            obs[i][0] = round(obs[i][0] * 52) + 52
            obs[i][1] = round(obs[i][1] * 33) + 34

        obs = obs.astype(np.int32)

        ind = 0
        ret[obs[0][0], obs[0][1], ind] = 255
        ind += 1

        if add_velocity:
            ret[obs[0][0], obs[0][1], ind] = 255
            ret[obs[0][0], obs[0][1], ind + 1] = 255
            ind += 2

        for i in range(1, number_players + 1):
            ret[obs[i][0], obs[i][1], ind] = 255
            if add_velocity:
                ret[obs[i][0], obs[i][1], ind + 1] = 255
                ret[obs[i][0], obs[i][1], ind + 2] = 255

        ind += 3 if add_velocity else 1

        for i in range(number_players + 1, 2 * number_players + 1):
            ret[obs[i][0], obs[i][1], ind] = 255
            if add_velocity:
                ret[obs[i][0], obs[i][1], ind + 1] = 255
                ret[obs[i][0], obs[i][1], ind + 2] = 255

        ind += 3 if add_velocity else 1

        return ret

    def reset(self):
        obs = self.env.reset()
        # print(obs)
        if self.modify_to_distances:
            obs = self.get_distances(obs, number_players=self.number_players)
        if self.modify_to_field_image and not self.modify_to_distances:
            obs = self.get_field_image(obs, number_players=self.number_players)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        if self.modify_to_distances:
            obs = self.get_distances(obs, number_players=self.number_players)
        if self.modify_to_field_image and not self.modify_to_distances:
            obs = self.get_field_image(obs, number_players=self.number_players)
        return obs, float(reward), done, info


if __name__ == "__main__":
    print("\n########################################\n")
    print(f"Observation Space for Environment: {env.observation_space.shape}")
    print(f"Action Space for Environment:      {env.action_space.shape}\n")

    print("Checking Environment...")
    check_env(env=RoboSoccerGym(
        env_name="Robosoccer-v1", n_players=10), warn=True)
    print("Environment Check Passed.\n")

    n_players = 10
    observation_size = (3 * (2 * n_players)) + 3 * 2
    total_timesteps = int(2 * (10 ** 4))
    policy_kwargs = {}
    policy_kwargs["optimizer_class"] = RMSpropTFLike
    policy_kwargs["optimizer_kwargs"] = dict(
        alpha=0.99, eps=1e-5, weight_decay=0)
    policy_kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64])]

    env = RoboSoccerGym(env_name="Robosoccer-v1", n_players=n_players)
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=2)

    print(f"\nNumber of Players:               {n_players}")
    print(f"Modified Observation Space:        {observation_size}")
    print(f"A2C Training Timesteps:            {total_timesteps}")
    print(f"Model Policy Network:            \n{model.policy}\n")
    print()
    print("Training started...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.\n")

    print("Testing...")

    iterations = 30
    rewards = []
    for _ in tqdm(range(iterations)):
        total_reward = get_score(env, model)
        rewards.append(total_reward)
    print(
        f"Avg Total Reward:                  {sum(rewards) / len(rewards)}\n")

    print(f"Saving Model to {MODEL_SAVE_LOCATION}")
    model.save(MODEL_SAVE_LOCATION)
    print("Model Saved.")
