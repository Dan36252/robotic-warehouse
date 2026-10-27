import main
import torch, time
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

observation_dict = main.env.reset()
with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    for t in range(500):
        main.env.render()
        time.sleep(0.25)
        observation_dict = main.policy(observation_dict)
        observation_dict = main.env.step(observation_dict)
        if observation_dict["next", "done"].item():
            break
        observation_dict = step_mdp(observation_dict)