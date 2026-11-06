import argparse
import time
from collections import deque

import numpy as np
import torch
import wandb

from env_np import UAVEnv
from ppo import Actor, Critic, RolloutBuffer, collect_rollout, compute_gae, ppo_update

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
n_envs, n_steps = 4, 1024

envs = [UAVEnv(seed=args.seed * 100 + i) for i in range(n_envs)]
obs = np.zeros((n_envs, 43), dtype=np.float32)
for i, env in enumerate(envs):
    obs[i], _ = env.reset()
    env.ep_return = 0.0

actor, critic = Actor(), Critic()
actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
buf = RolloutBuffer(n_steps, n_envs, 43, 2)

run = wandb.init(project="uav-drl", name=f"PPO_scratch_seed{args.seed}", config=vars(args))

recent = deque(maxlen=100)   # last 100 episodes, for success rate
total = 0
start = time.time()
n_iters = args.steps // (n_steps * n_envs)

for it in range(1, n_iters + 1):
    obs, finished = collect_rollout(envs, obs, actor, critic, buf)
    recent.extend(finished)
    total += n_steps * n_envs

    with torch.no_grad():
        last_values = critic(torch.tensor(obs)).numpy()
    advantages, returns = compute_gae(buf, last_values)
    stats = ppo_update(actor, critic, actor_opt, critic_opt, buf, advantages, returns)

    events = [e["event"] for e in recent]
    log = {
        "rollout/success_rate": events.count("goal") / max(len(events), 1),
        "rollout/collision_rate": events.count("collision") / max(len(events), 1),
        "rollout/ep_rew_mean": np.mean([e["return"] for e in recent]) if recent else 0,
        "rollout/ep_len_mean": np.mean([e["length"] for e in recent]) if recent else 0,
        "time/fps": total / (time.time() - start),
        "global_step": total,
    }
    log.update({f"train/{k}": v for k, v in stats.items()})
    wandb.log(log, step=total)

    print(f"iter {it:3d} | steps {total:7d} | success {log['rollout/success_rate']:.2f} | "
          f"reward {log['rollout/ep_rew_mean']:7.1f} | kl {stats['approx_kl']:.4f} | "
          f"ev {stats['explained_variance']:.2f} | std {stats['std']:.2f}")

torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
           f"ppo_scratch_seed{args.seed}.pt")
print("saved model")
run.finish()