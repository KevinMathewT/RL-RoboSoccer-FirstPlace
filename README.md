# AI RoboSoccer First Place Solution

This repository contains the code and a writeup for the first place solution of the AI RoboSoccer Competition, which was the the flagship event conducted by IEEE Student Chapter, Birla Institute of Technology & Science (BITS), Pilani.

The event works on a custom built environment for simulation, made using OpenAI Gym. They had developed an environment on Gym using Pygame with its assets including the defenders and the attackers. Furthermore, the feedback of every action and the state-space log would be available via pre-defined functions. 

Find more about the competition [here](https://dare2compete.com/o/ai-robosoccer-apogee-bits-pilani-birla-institute-of-technology-science-bits-pilani-151055), and find the repo for the OpenAI Gym environment [here](https://github.com/IEEE-BITS-Pilani-Student-Chapter/robo-soccer).

![random_actions](https://user-images.githubusercontent.com/22857545/114004006-df51a000-987b-11eb-82b3-6dfb2456854b.gif)

## Solution Outline
### Observation Space
The original observation space from the gym environment provides the x and y coordinates of each player from both teams and the ball, along with the velocities of each player.

For my solution I modified the observation space to be the manhattan distance and the direction vector of the ball from each player and both the goal posts, and dropped the velocities completely. This modification was crucial for the winning solution.

### Reinforcement Learning Agent
For the RL Agent, I used an Advantage Actor Critic (A2C) network, implemented in [StableBaselines3](https://github.com/DLR-RM/stable-baselines3). 

```python
ActorCriticPolicy(
  (features_extractor): FlattenExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (mlp_extractor): MlpExtractor(
    (shared_net): Sequential()
    (policy_net): Sequential(
      (0): Linear(in_features=18, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=18, out_features=64, bias=True)
      (1): Tanh()
    )
  )
  (action_net): Linear(in_features=64, out_features=20, bias=True)
  (value_net): Linear(in_features=64, out_features=1, bias=True)
)
```
For training I used the `RMSpropTFLike` optimizer, as it stabilized training as given [here](https://github.com/DLR-RM/stable-baselines3/pull/110#issuecomment-663255241).
```python
policy_kwargs["optimizer_class"] = RMSpropTFLike
policy_kwargs["optimizer_kwargs"] = dict(
    alpha=0.99, eps=1e-5, weight_decay=0)
```

## Prerequisites
Use pip to install requirements for the repository:
```
$ pip install -r requirements.txt
```

## Training
For training the model execute:
```bash
$ python -m src.2v2.train
```
Similarly you can execute `python -m src.5v5.train` and `python -m src.10v10.train` to train the agent for 5v5 and 10v10 environments.

Models are saved in `./generated/`

## Experiments
Average Scores over 30 Episodes for my experiments in the 2v2 environment:
| Model | Observation Space | MLP Extractor Policy Network | MLP Extractor Value Network | Training Timsteps | Score |
| ----- |:-----------------:|:----------------------------:|:---------------------------:|:--------------------:|:-----------:|
| A2C   | Distance + Direction Vector of left team from ball | 2 Layered (64, 64)  | 2 Layered (64, 64) | 10000 | 2617.0491 |
| A2C   | Distance + Direction Vector of left team from ball | 2 Layered (64, 64)  | 2 Layered (64, 64) | 15000 | 3033.4046 |
| A2C   | Distance + Direction Vector of left team from ball | 2 Layered (64, 64)  | 2 Layered (64, 64) | 20000 | 2148.8943 |
| A2C   | Distance + Direction Vector of both teams from ball | 2 Layered (64, 64)  | 2 Layered (64, 64) | 15000 | 2148.8943 |
| A2C   | Distance + Direction Vector of both teams from ball + Ball from both goals | 2 Layered (64, 64)  | 2 Layered (64, 64) | 15000 | 3496.6428 |
| A2C   | Distance + Direction Vector of both teams from ball + Ball from both goals | 2 Layered (64, 64)  | 2 Layered (64, 64) | 20000 | 5306.7699 |
| A2C   | Distance + Direction Vector of both teams from ball + Ball from both goals | 2 Layered (64, 64)  | 1 Layered (64) | 15000 | 5471.0487 |
| A2C   | Distance + Direction Vector of both teams from ball + Ball from both goals | 2 Layered (64, 64)  | 1 Layered (64) | 10000 | 5700.8237 |

## Rendering
You can also render and watch a game played by the trained agent by executing:
```bash
$ python -m src.2v2.render_game
```
Similarly you can execute `python -m src.5v5.render_game` and `python -m src.10v10.render_game` to render a game played in the 5v5 and 10v10 environments.
