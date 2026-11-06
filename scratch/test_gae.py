import numpy as np
import torch
from env_np import UAVEnv
from ppo import Actor, Critic, RolloutBuffer, collect_rollout, compute_gae

# 1. tiny example: 3 steps, 1 env, the episode ends at the last step
buf = RolloutBuffer(3, 1, 43, 2)
buf.rewards[:, 0] = [1.0, 1.0, 1.0]
buf.values[:, 0] = [0.5, 0.5, 0.5]
buf.dones[:, 0] = [0.0, 0.0, 1.0]
adv, ret = compute_gae(buf, last_values=np.array([10.0]))
print("advantages:", adv[:, 0].round(4))   # expected [2.3731 1.4652 0.5]
print("returns:   ", ret[:, 0].round(4))   # expected [2.8731 1.9652 1.0]

# 2. real rollout
envs = [UAVEnv(seed=i) for i in range(4)]
obs = np.zeros((4, 43), dtype=np.float32)
for i, env in enumerate(envs):
    obs[i], _ = env.reset()
    env.ep_return = 0.0

actor, critic = Actor(), Critic()
buf = RolloutBuffer(1024, 4, 43, 2)
obs, finished = collect_rollout(envs, obs, actor, critic, buf)
with torch.no_grad():
    last_values = critic(torch.tensor(obs)).numpy()

adv, ret = compute_gae(buf, last_values)
print("advantage shape:", adv.shape)
print(f"advantage mean {adv.mean():.2f}, std {adv.std():.2f}")
print(f"returns min {ret.min():.1f}, max {ret.max():.1f}")