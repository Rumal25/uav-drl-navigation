import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
DT = 0.25
MAX_STEPS = 600
UAV_RADIUS = 0.5
GOAL_RADIUS = 1.5
MAP_SIZE = 50.0      # metres
N_OBSTACLES = 15
N_RAYS = 36
MAX_RANGE = 10.0
STEP = 0.1
V_MAX = 3.0
W_MAX = 1.5
DIAG = np.sqrt(2) * MAP_SIZE          # 70.7 m, the longest possible distance


rng = np.random.default_rng(seed=0)   # random number generator (seed = same map each run)

# ---------- 1. place obstacles ----------
obstacles = []                       # each item: [x, y, radius]
while len(obstacles) < N_OBSTACLES:
    r = rng.uniform(1.0, 3.0)                          # random radius 1-3 m
    x, y = rng.uniform(r + 2, MAP_SIZE - r - 2, size=2)  # keep 2 m from the walls
    # check: is this new circle far enough from all old circles?
    ok = True
    for (ox, oy, orad) in obstacles:
        dist = np.hypot(x - ox, y - oy)               # distance between centres
        if dist < r + orad + 2.0:                     # 2 m gap between circles
            ok = False
            break
    if ok:
        obstacles.append([x, y, r])
obstacles = np.array(obstacles)

# ---------- 2. helper: how far is a point from the nearest obstacle edge? ----------
def clearance(p):
    d = np.hypot(obstacles[:, 0] - p[0], obstacles[:, 1] - p[1]) - obstacles[:, 2]
    return d.min()

# ---------- 3. pick start and goal ----------
while True:
    start = rng.uniform(3, MAP_SIZE - 3, size=2)
    goal = rng.uniform(3, MAP_SIZE - 3, size=2)
    far_apart = np.linalg.norm(goal - start) > 25.0
    free = clearance(start) > 2.0 and clearance(goal) > 2.0
    if far_apart and free:
        break

#print("obstacles:\n", np.round(obstacles, 1))
print("start:", np.round(start, 1), " goal:", np.round(goal, 1))

def lidar(x, y, yaw, obstacles):
    """Return 36 distances (m) and the 36 ray angles."""
    angles = yaw + np.arange(N_RAYS) * (2 * np.pi / N_RAYS)   # ray 0 = straight ahead
    dists = np.arange(STEP, MAX_RANGE + STEP, STEP)            # 0.1, 0.2, ... 10.0  (100 points)

    # coordinates of every test point: shape (36 rays, 100 points)
    px = x + np.cos(angles)[:, None] * dists[None, :]
    py = y + np.sin(angles)[:, None] * dists[None, :]

    # is each point inside any obstacle?   shape (36, 100, N_OBSTACLES) -> (36, 100)
    dx = px[:, :, None] - obstacles[:, 0]
    dy = py[:, :, None] - obstacles[:, 1]
    inside_obstacle = (dx**2 + dy**2 <= obstacles[:, 2]**2).any(axis=2)

    # is each point outside the map (a wall)?
    outside_map = (px < 0) | (px > MAP_SIZE) | (py < 0) | (py > MAP_SIZE)

    hit = inside_obstacle | outside_map                # (36, 100) True where something is

    ranges = np.full(N_RAYS, MAX_RANGE)                # default: nothing seen -> 10 m
    has_hit = hit.any(axis=1)                          # which rays hit something
    first_hit = hit.argmax(axis=1)                     # index of the first True on each ray
    ranges[has_hit] = dists[first_hit[has_hit]]
    return ranges, angles


def get_obs(x, y, yaw, v, w, goal, obstacles):
    # goal distance and relative bearing
    dx, dy = goal[0] - x, goal[1] - y
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx) - yaw          # angle to goal, relative to the drone

    ranges, _ = lidar(x, y, yaw, obstacles)

    obs = np.concatenate([
        [dist / DIAG,                           # 0
         np.sin(bearing), np.cos(bearing),      # 1, 2
         v / V_MAX,                             # 3
         w / W_MAX,                             # 4
         x / MAP_SIZE * 2 - 1,                  # 5
         y / MAP_SIZE * 2 - 1],                 # 6
        ranges / MAX_RANGE                      # 7 ... 42
    ])
    return obs.astype(np.float32)


def clearance_with_walls(x, y):
    return min(clearance((x, y)), x, y, MAP_SIZE - x, MAP_SIZE - y)

def run_episode(policy, title):
    x, y = start
    yaw = rng.uniform(-np.pi, np.pi)             # random starting direction
    v, w = 0.0, 0.0
    prev_dist = np.hypot(goal[0] - x, goal[1] - y)
    total_reward, path = 0.0, [(x, y)]

    for step in range(MAX_STEPS):
        obs = get_obs(x, y, yaw, v, w, goal, obstacles)
        a = policy(obs)                          # 2 numbers in [-1, 1]

        # --- 1.5 action -> command ---
        v = (a[0] + 1) / 2 * V_MAX
        w = a[1] * W_MAX

        # --- move (from Step 1.2) ---
        yaw = (yaw + w * DT + np.pi) % (2 * np.pi) - np.pi
        x += v * np.cos(yaw) * DT
        y += v * np.sin(yaw) * DT
        path.append((x, y))

        # --- 1.6 reward ---
        dist = np.hypot(goal[0] - x, goal[1] - y)
        clear = clearance_with_walls(x, y)
        r = 10 * (prev_dist - dist) - 0.05 - 0.05 * abs(a[1])
        if clear < 2.0:
            r -= 0.5 * (2.0 - clear)
        prev_dist = dist

        terminated, truncated, event = False, False, "running"
        if clear < UAV_RADIUS:
            r -= 100; terminated = True; event = "collision"
        elif dist < GOAL_RADIUS:
            r += 100; terminated = True; event = "goal"
        elif step + 1 >= MAX_STEPS:
            truncated = True; event = "timeout"

        total_reward += r
        if terminated or truncated:
            break

    print(f"{title}: {event} after {step + 1} steps, total reward = {total_reward:.1f}")
    return np.array(path), event

# --- two test "pilots" (no learning yet) ---
def random_policy(obs):
    return rng.uniform(-1, 1, size=2)

def simple_policy(obs):
    # turn toward the goal: obs[1] = sin(bearing). Positive -> goal is left -> turn left.
    return np.array([0.0, np.clip(2 * obs[1], -1, 1)])   # half speed, steer to goal

path_r, ev_r = run_episode(random_policy, "random pilot")
path_s, ev_s = run_episode(simple_policy, "simple pilot")

fig, ax = plt.subplots(figsize=(6, 6))
for (ox, oy, rad) in obstacles:
    ax.add_patch(Circle((ox, oy), rad, color="gray"))
ax.plot(path_r[:, 0], path_r[:, 1], label=f"random ({ev_r})")
ax.plot(path_s[:, 0], path_s[:, 1], label=f"simple ({ev_s})")
ax.plot(*start, "go", markersize=10); ax.plot(*goal, "r*", markersize=15)
ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect("equal")
ax.legend(); plt.show()