import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from uav_env import UAVNavEnv

def simple_policy(obs):
    return np.array([0.0, np.clip(2 * obs[1], -1, 1)])   # steer toward the goal, ignore LiDAR

def fly(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        action = policy(obs) if policy else env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total += r
        done = terminated or truncated
    return info["event"], total

def draw(ax, env, title):
    for (ox, oy, r) in env.obstacles:
        ax.add_patch(Circle((ox, oy), r, color="gray"))
    p = np.array(env.path)
    ax.plot(p[:, 0], p[:, 1], "b-", linewidth=1.5)
    # LiDAR at the final position
    ranges = env._lidar()
    angles = env.yaw + np.arange(env.N_RAYS) * (2 * np.pi / env.N_RAYS)
    for rg, ang in zip(ranges, angles):
        ax.plot([env.pos[0], env.pos[0] + rg * np.cos(ang)],
                [env.pos[1], env.pos[1] + rg * np.sin(ang)],
                color="red" if rg < env.MAX_RANGE else "orange", linewidth=0.5)
    ax.plot(*p[0], "go", markersize=8)
    ax.plot(*env.goal, "r*", markersize=14)
    ax.add_patch(Circle(env.goal, env.GOAL_RADIUS, fill=False, color="red", linestyle="--"))
    ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)

env = UAVNavEnv()

# ---- 1. many episodes: count how they end ----
for name, pol in [("random", None), ("simple", simple_policy)]:
    events = [fly(env, pol, seed)[0] for seed in range(50)]
    print(f"{name:7s}: goal={events.count('goal')}, collision={events.count('collision')}, "
          f"timeout={events.count('timeout')}  (out of 50)")

# ---- 2. draw 6 example flights ----
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for i, ax in enumerate(axes.flat):
    pol, name = (None, "random") if i < 3 else (simple_policy, "simple")
    event, total = fly(env, pol, seed=i)
    draw(ax, env, f"{name}, seed {i}: {event}, reward {total:.0f}, {env.steps} steps")
plt.tight_layout(); plt.show()