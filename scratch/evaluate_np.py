import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from env_np import UAVEnv
from ppo import Actor

test_seeds = range(1000, 1200)   # same 200 maps as the SB3 evaluation

actor = Actor()
actor.load_state_dict(torch.load("ppo_scratch_seed0.pt")["actor"])
actor.eval()


def policy(obs):
    # deterministic: use the mean action, no sampling
    with torch.no_grad():
        return actor(torch.tensor(obs)).mean.numpy()


def fly(env, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
    path = np.array(env.path)
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    straight = np.linalg.norm(env.goal - path[0])
    return info["event"], env.steps * env.dt, length / straight


for n_obs in [15, 25]:
    env = UAVEnv(n_obstacles=n_obs)
    results = [fly(env, s) for s in test_seeds]
    events = [r[0] for r in results]
    ok = [r for r in results if r[0] == "goal"]
    n = len(events)
    print(f"{n_obs} obstacles: success {100 * events.count('goal') / n:.1f}%, "
          f"collision {100 * events.count('collision') / n:.1f}%, "
          f"timeout {100 * events.count('timeout') / n:.1f}%, "
          f"time {np.mean([r[1] for r in ok]):.1f} s, "
          f"path ratio {np.mean([r[2] for r in ok]):.2f}")


# trajectories on the same 4 maps as before
env = UAVEnv()
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, seed in zip(axes, [1000, 1001, 1002, 1003]):
    event, t, _ = fly(env, seed)
    for (ox, oy, r) in env.obstacles:
        ax.add_patch(Circle((ox, oy), r, color="gray"))
    p = np.array(env.path)
    ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2)
    ax.plot(*p[0], "go", markersize=8)
    ax.plot(*env.goal, "r*", markersize=14)
    ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect("equal")
    ax.set_title(f"map {seed}: {event} ({t:.0f} s)", fontsize=10)
plt.suptitle("PPO from scratch on unseen maps")
plt.tight_layout()
plt.savefig("trajectories_scratch.png", dpi=150)
plt.show()