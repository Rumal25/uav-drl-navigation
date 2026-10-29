import numpy as np
import matplotlib.pyplot as plt

DT = 0.25        # seconds per step
V_MAX = 3.0      # max forward speed (m/s)
W_MAX = 1.5      # max turn rate (rad/s)

def move(x, y, yaw, v, w):
    """Move the drone one time step with speed v and turn rate w."""
    v = np.clip(v, 0.0, V_MAX)          # limit speed: no going backwards
    w = np.clip(w, -W_MAX, W_MAX)       # limit turning
    yaw = yaw + w * DT
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi   # keep angle between -pi and pi
    x = x + v * np.cos(yaw) * DT
    y = y + v * np.sin(yaw) * DT
    return x, y, yaw

# try 3 different constant commands for 80 steps (20 s)
commands = {"straight (v=2, w=0)": (2.0, 0.0),
            "gentle left (v=2, w=0.3)": (2.0, 0.3),
            "sharp right (v=2, w=-1.0)": (2.0, -1.0)}

plt.figure(figsize=(6, 6))
for name, (v, w) in commands.items():
    x, y, yaw = 25.0, 10.0, np.pi / 2      # start in the middle-bottom, facing up
    path = [(x, y)]
    for step in range(80):
        x, y, yaw = move(x, y, yaw, v, w)
        path.append((x, y))
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], label=name)

plt.plot(25, 10, "go", markersize=10, label="start")
plt.xlim(0, 50); plt.ylim(0, 50); plt.gca().set_aspect("equal")
plt.legend(); plt.title("Drone motion with constant commands"); plt.show()