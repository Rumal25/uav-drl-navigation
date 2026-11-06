import torch
from env_np import UAVEnv
from ppo import Actor, Critic

actor, critic = Actor(), Critic()

n_actor = sum(p.numel() for p in actor.parameters())
n_critic = sum(p.numel() for p in critic.parameters())
print("actor parameters:", n_actor)     # should be 77,572
print("critic parameters:", n_critic)   # should be 77,313

# one real observation from the environment
env = UAVEnv(seed=0)
obs, _ = env.reset()
obs_t = torch.tensor(obs).unsqueeze(0)   # shape (1, 43), a batch of one

dist = actor(obs_t)
action = dist.sample()
log_prob = dist.log_prob(action).sum(-1)   # add the two actions' log probs
value = critic(obs_t)

print("mean:", dist.mean)
print("std:", dist.stddev)
print("action:", action, "log prob:", log_prob)
print("value:", value)