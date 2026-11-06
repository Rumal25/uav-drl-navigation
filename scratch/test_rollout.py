import time
import numpy as np
from env_np import UAVEnv
from ppo import Actor, Critic, RolloutBuffer, collect_rollout

n_envs, n_steps = 4, 1024
envs = [UAVEnv(seed=i) for i in range(n_envs)]
obs = np.zeros((n_envs, 43), dtype=np.float32)
for i, env in enumerate(envs):
    obs[i], _ = env.reset()
    env.ep_return = 0.0

actor, critic = Actor(), Critic()
buf = RolloutBuffer(n_steps, n_envs, 43, 2)

start = time.time()
obs, finished = collect_rollout(envs, obs, actor, critic, buf)
print(f"collected {n_steps * n_envs} steps in {time.time() - start:.1f} s")

print("obs:", buf.obs.shape, " actions:", buf.actions.shape, " rewards:", buf.rewards.shape)
print("episodes finished:", len(finished))
events = [e["event"] for e in finished]
print("results:", {k: events.count(k) for k in set(events)})
print("mean episode return:", np.mean([e["return"] for e in finished]))
print("dones in buffer:", int(buf.dones.sum()))