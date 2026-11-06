import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer(in_dim, out_dim, gain=np.sqrt(2)):
    # orthogonal init
    lin = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class Actor(nn.Module):
    def __init__(self, obs_dim=43, act_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, act_dim, gain=0.01),   # small gain so the first actions are near 0
        )
        # one std per action, learned, same in every state
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, obs_dim=43, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            layer(obs_dim, hidden), nn.Tanh(),
            layer(hidden, hidden), nn.Tanh(),
            layer(hidden, 1, gain=1.0),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)   # shape (batch,) instead of (batch, 1)

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, act_dim):
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)


def collect_rollout(envs, obs, actor, critic, buf, gamma=0.99):
    n_steps = buf.obs.shape[0]
    finished = []

    for t in range(n_steps):
        obs_t = torch.tensor(obs)
        with torch.no_grad():
            dist = actor(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            value = critic(obs_t)

        buf.obs[t] = obs
        buf.actions[t] = action.numpy()
        buf.log_probs[t] = log_prob.numpy()
        buf.values[t] = value.numpy()

        next_obs = np.zeros_like(obs)
        for i, env in enumerate(envs):
            o, r, terminated, truncated, info = env.step(action[i].numpy())
            env.ep_return += r

            # time limit is not a real end, so add the value of where we stopped
            if truncated and not terminated:
                with torch.no_grad():
                    r += gamma * critic(torch.tensor(o)).item()

            buf.rewards[t, i] = r
            buf.dones[t, i] = float(terminated or truncated)

            if terminated or truncated:
                finished.append({"event": info["event"], "return": env.ep_return, "length": env.steps})
                o, _ = env.reset()
                env.ep_return = 0.0
            next_obs[i] = o
        obs = next_obs

    return obs, finished

def compute_gae(buf, last_values, gamma=0.99, lam=0.95):
    """Advantages with GAE, computed backwards through the rollout."""
    n_steps = buf.rewards.shape[0]
    advantages = np.zeros_like(buf.rewards)
    gae = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_values = last_values          # value of the state after the rollout
        else:
            next_values = buf.values[t + 1]
        not_done = 1.0 - buf.dones[t]

        delta = buf.rewards[t] + gamma * next_values * not_done - buf.values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + buf.values
    return advantages, returns   

def explained_variance(values, returns):
    # 1 = critic predicts returns perfectly, 0 = no better than the mean
    var = np.var(returns)
    return np.nan if var == 0 else 1 - np.var(returns - values) / var


def ppo_update(actor, critic, actor_opt, critic_opt, buf, advantages, returns,
               epochs=10, batch_size=256, clip=0.2, max_grad_norm=0.5):
    # flatten (steps, envs, ...) into one long list of samples
    obs = torch.tensor(buf.obs.reshape(-1, buf.obs.shape[-1]))
    actions = torch.tensor(buf.actions.reshape(-1, buf.actions.shape[-1]))
    old_log_probs = torch.tensor(buf.log_probs.reshape(-1))
    adv_all = torch.tensor(advantages.reshape(-1))
    ret_all = torch.tensor(returns.reshape(-1))

    n = obs.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "approx_kl": [], "clip_fraction": []}

    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            dist = actor(obs[b])
            new_log_probs = dist.log_prob(actions[b]).sum(-1)
            log_ratio = new_log_probs - old_log_probs[b]
            ratio = log_ratio.exp()

            adv = adv_all[b]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # clipped surrogate, negative because the optimizer minimizes
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            actor_opt.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_opt.step()

            value_loss = ((critic(obs[b]) - ret_all[b]) ** 2).mean()

            critic_opt.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > clip).float().mean().item()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_fraction"].append(clip_frac)

    stats = {k: float(np.mean(v)) for k, v in stats.items()}
    stats["explained_variance"] = explained_variance(buf.values.reshape(-1), returns.reshape(-1))
    stats["std"] = actor.log_std.exp().mean().item()
    return stats