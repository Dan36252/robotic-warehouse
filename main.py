# This module, when run, gets the "wrapped" RWARE environment (using torchrl_wrapper.py),
# and using the hyperparemeters defined in this file, starts training a Policy using RL
# to maximize the reward.

from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from rware.torchrl_wrapper import IndependentMultiCategorical, make_centralized_env
from rware.warehouse import RewardType

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- environment ---
N_AGENTS = 3
MSG_BITS = 0

env, nvec = make_centralized_env(
   3, 3, 1, N_AGENTS, MSG_BITS, 1, 4, None, 500, RewardType.TWO_STAGE, device=device
)
obs_dim = env.observation_spec["observation"].shape[-1]
action_dim = sum(nvec)

# --- policy and value networks ---
actor_net = MLP(in_features=obs_dim, out_features=action_dim, depth=2, num_cells=256, device=device)
policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=IndependentMultiCategorical,
    distribution_kwargs={"nvec": nvec},
    return_log_prob=True,
)

critic_net = MLP(in_features=obs_dim, out_features=1, depth=2, num_cells=256, device=device)
critic = ValueOperator(module=critic_net, in_keys=["observation"])

# --- PPO training setup ---
FRAMES_PER_BATCH = 4000
TOTAL_FRAMES = 5_000_000
NUM_EPOCHS = 10
MINIBATCH_SIZE = 400
CLIP_EPSILON = 0.2
GAMMA = 0.99
LMBDA = 0.95
ENTROPY_COEF = 0.01
LR = 3e-4
MAX_GRAD_NORM = 1.0
CHECKPOINT_EVERY = 50
CHECKPOINT_DIR = Path("checkpoints4")

collector = Collector(
    env, policy, frames_per_batch=FRAMES_PER_BATCH, total_frames=TOTAL_FRAMES, device=device
)

advantage_module = GAE(gamma=GAMMA, lmbda=LMBDA, value_network=critic, average_gae=True)

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=CLIP_EPSILON,
    entropy_bonus=True,
    entropy_coeff=ENTROPY_COEF,
    normalize_advantage=True,
)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(FRAMES_PER_BATCH, device=device),
    sampler=SamplerWithoutReplacement(),
    batch_size=MINIBATCH_SIZE,
)

optim = torch.optim.Adam(loss_module.parameters(), lr=LR)

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

start_batch = 0
checkpoints = sorted(
    CHECKPOINT_DIR.glob("batch_*.pt"), key=lambda p: int(p.stem.split("_")[1])
)
if checkpoints:
    latest = checkpoints[-1]
    ckpt = torch.load(latest, map_location=device)
    actor_net.load_state_dict(ckpt["actor"])
    critic_net.load_state_dict(ckpt["critic"])
    start_batch = int(latest.stem.split("_")[1])
    print(f"Loaded checkpoint {latest}, resuming after batch {start_batch}")

if __name__ == "__main__":
    for batch_idx, data in enumerate(collector, start=start_batch):
        advantage_module(data)
        replay_buffer.extend(data.reshape(-1))

        objective_losses, critic_losses, entropy_losses = [], [], []
        for _ in range(NUM_EPOCHS):
            for _ in range(FRAMES_PER_BATCH // MINIBATCH_SIZE):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                optim.step()
                optim.zero_grad()

                objective_losses.append(loss_vals["loss_objective"].item())
                critic_losses.append(loss_vals["loss_critic"].item())
                entropy_losses.append(loss_vals["loss_entropy"].item())

        mean_reward = data["next", "reward"].mean().item()
        mean_loss_objective = sum(objective_losses) / len(objective_losses)
        mean_loss_critic = sum(critic_losses) / len(critic_losses)
        mean_loss_entropy = sum(entropy_losses) / len(entropy_losses)
        print(
            f"batch {batch_idx}: mean_reward={mean_reward:.4f} "
            f"loss_objective={mean_loss_objective:.4f} loss_critic={mean_loss_critic:.4f} "
            f"loss_entropy={mean_loss_entropy:.4f}"
        )

        # Record reward and loss components, each against its batch number
        with open(LOG_DIR / "training_reward.txt", "a") as log_file:
            log_file.write(f"{mean_reward}\n")
        with open(LOG_DIR / "training_batch_nums.txt", "a") as log_file:
            log_file.write(f"{batch_idx}\n")
        with open(LOG_DIR / "loss_objective.txt", "a") as log_file:
            log_file.write(f"{mean_loss_objective}\n")
        with open(LOG_DIR / "loss_critic.txt", "a") as log_file:
            log_file.write(f"{mean_loss_critic}\n")
        with open(LOG_DIR / "loss_entropy.txt", "a") as log_file:
            log_file.write(f"{mean_loss_entropy}\n")

        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            torch.save(
                {"actor": actor_net.state_dict(), "critic": critic_net.state_dict()},
                CHECKPOINT_DIR / f"batch_{batch_idx + 1}.pt",
            )

progress_log = """
checkpoints - The very first training run, with a huge warehouse and unoptimized parameters
checkpoints2 - Second training run, with same huge warehouse but slightly more optimized observation/action spaces
checkpoints3 - Third training run, tiny warehouse: env(1, 3, 1, 2, 0, 1, 2, None, 500, RewardType.TWO_STAGE)
checkpoints4 - Fourth training run, larger warehouse: env(3, 3, 1, 3, 0, 1, 4, None, 500, RewardType.TWO_STAGE)
"""