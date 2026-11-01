"""
Step 4: Evaluate the trained PPO and SAC policies on 200 NEW maps.
- deterministic actions (no exploration randomness)
- same maps for every pilot (fair comparison)
- baseline: the simple "steer to goal" pilot
- two difficulty levels: 15 obstacles (as in training) and 25 obstacles (harder, never seen)
Results are printed, saved as trajectories.png, and logged to W&B.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import wandb
from stable_baselines3 import PPO, SAC
from uav_env import UAVNavEnv

TEST_SEEDS = range(1000, 1200)        # 200 maps, never used in training

# ---------- W&B ----------
run = wandb.init(project="uav-drl", name="evaluation", job_type="eval")
all_results = wandb.Table(columns=["obstacles", "pilot", "success", "collision",
                                   "timeout", "time_s", "path_ratio"])

# ---------- the pilots ----------
ppo = PPO.load("models/PPO_seed0.zip", device="cpu")
sac = SAC.load("models/SAC_seed0.zip", device="cpu")
pilots = {
    "simple": lambda obs: np.array([0.0, np.clip(2 * obs[1], -1, 1)]),   # steer to goal, ignore LiDAR
    "PPO":    lambda obs: ppo.predict(obs, deterministic=True)[0],
    "SAC":    lambda obs: sac.predict(obs, deterministic=True)[0],
}


def fly(policy, env, seed):
    """Fly one episode on the map given by seed. Returns (event, flight time, path ratio)."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
    path = np.array(env.path)
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))   # flown distance
    straight = np.linalg.norm(env.goal - path[0])                     # straight-line distance
    return info["event"], env.steps * env.DT, length / straight


def evaluate(n_obstacles):
    """Test every pilot on the same 200 maps; print and log the results."""
    env = UAVNavEnv(n_obstacles=n_obstacles)
    table = wandb.Table(columns=["pilot", "success"])

    print(f"\n=== {n_obstacles} obstacles, {len(TEST_SEEDS)} new maps ===")
    print(f"{'pilot':8s} {'success':>8s} {'collision':>10s} {'timeout':>8s} "
          f"{'time(s)':>8s} {'path ratio':>11s}")

    for name, policy in pilots.items():
        res = [fly(policy, env, s) for s in TEST_SEEDS]
        ev = [r[0] for r in res]
        ok = [r for r in res if r[0] == "goal"]
        n = len(ev)
        success = 100 * ev.count("goal") / n
        collision = 100 * ev.count("collision") / n
        timeout = 100 * ev.count("timeout") / n
        t = float(np.mean([r[1] for r in ok])) if ok else float("nan")   # only successful flights
        p = float(np.mean([r[2] for r in ok])) if ok else float("nan")

        print(f"{name:8s} {success:7.1f}% {collision:9.1f}% {timeout:7.1f}% {t:8.1f} {p:11.2f}")
        all_results.add_data(n_obstacles, name, success, collision, timeout, t, p)
        table.add_data(name, success)

    wandb.log({f"eval/success_{n_obstacles}_obstacles":
               wandb.plot.bar(table, "pilot", "success",
                              title=f"Success rate (%), {n_obstacles} obstacles, 200 new maps")})


evaluate(15)    # same difficulty as training
evaluate(25)    # harder: never seen in training

# ---------- trajectory pictures: PPO (top) vs SAC (bottom), same 4 maps ----------
env = UAVNavEnv()
seeds = [1000, 1001, 1002, 1003]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
for row, name in enumerate(["PPO", "SAC"]):
    for col, s in enumerate(seeds):
        ax = axes[row, col]
        event, t, _ = fly(pilots[name], env, s)
        for (ox, oy, r) in env.obstacles:
            ax.add_patch(Circle((ox, oy), r, color="gray"))
        p = np.array(env.path)
        ax.plot(p[:, 0], p[:, 1], "b-", linewidth=2)
        ax.plot(*p[0], "go", markersize=8)                 # start
        ax.plot(*env.goal, "r*", markersize=14)            # goal
        ax.add_patch(Circle(env.goal, env.GOAL_RADIUS, fill=False, color="red", linestyle="--"))
        ax.set_xlim(0, 50); ax.set_ylim(0, 50); ax.set_aspect("equal")
        ax.set_title(f"{name}, map {s}: {event} ({t:.0f} s)", fontsize=10)
plt.tight_layout()
plt.savefig("trajectories.png", dpi=150)

# ---------- send everything to W&B ----------
wandb.log({
    "eval/results": all_results,
    "eval/trajectories": wandb.Image("trajectories.png"),
})
run.finish()

plt.show()    