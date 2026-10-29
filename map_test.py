import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

MAP_SIZE = 50.0      # metres
N_OBSTACLES = 15

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

print("obstacles:\n", np.round(obstacles, 1))
print("start:", np.round(start, 1), " goal:", np.round(goal, 1))

# ---------- 4. draw it ----------
fig, ax = plt.subplots(figsize=(6, 6))
for (x, y, r) in obstacles:
    ax.add_patch(Circle((x, y), r, color="gray"))
ax.plot(*start, "go", markersize=10, label="start")
ax.plot(*goal, "r*", markersize=15, label="goal")
ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect("equal")
ax.legend(); ax.set_title("50 m x 50 m map")
plt.show()