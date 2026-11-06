import numpy as np


class UAVEnv:
    """2D UAV navigation in a 50 x 50 m map with circular obstacles."""

    def __init__(self, n_obstacles=15, gps_noise=0.0, gps_offset=0.0, seed=None):
        self.map_size = 50.0
        self.n_obstacles = n_obstacles

        # lidar
        self.n_rays = 36
        self.max_range = 10.0
        self.ray_step = 0.1

        # uav limits
        self.v_max = 3.0
        self.w_max = 1.5
        self.dt = 0.25
        self.max_steps = 600
        self.uav_radius = 0.5
        self.goal_radius = 1.5
        self.diag = np.sqrt(2) * self.map_size

        # gps error, only used for robustness tests
        self.gps_noise = gps_noise
        self.gps_offset = gps_offset

        self.obs_dim = 7 + self.n_rays   # 43
        self.act_dim = 2                 # forward speed, yaw rate

        self.rng = np.random.default_rng(seed)

    def _make_map(self):
        rng, L = self.rng, self.map_size
        while True:
            obstacles = []
            for _ in range(1000):
                if len(obstacles) == self.n_obstacles:
                    break
                r = rng.uniform(1.0, 3.0)
                c = rng.uniform(r + 2, L - r - 2, size=2)
                # keep at least 2 m between obstacles
                if all(np.hypot(*(c - o[:2])) > r + o[2] + 2.0 for o in obstacles):
                    obstacles.append(np.array([c[0], c[1], r]))
            if len(obstacles) < self.n_obstacles:
                continue  # map too crowded, try again
            self.obstacles = np.array(obstacles)

            for _ in range(1000):
                start = rng.uniform(3, L - 3, size=2)
                goal = rng.uniform(3, L - 3, size=2)
                far_enough = np.linalg.norm(goal - start) > 25.0
                free = self._clearance(start) > 2.0 and self._clearance(goal) > 2.0
                if far_enough and free:
                    return start, goal

    def _clearance(self, p):
        # distance from point p to the nearest obstacle edge
        d = np.hypot(self.obstacles[:, 0] - p[0], self.obstacles[:, 1] - p[1])
        return (d - self.obstacles[:, 2]).min()

    def _clearance_with_walls(self, p):
        L = self.map_size
        return min(self._clearance(p), p[0], p[1], L - p[0], L - p[1])

    def _lidar(self):
        x, y = self.pos
        angles = self.yaw + np.arange(self.n_rays) * (2 * np.pi / self.n_rays)
        dists = np.arange(self.ray_step, self.max_range + self.ray_step, self.ray_step)

        # points along every ray, shape (36, 100)
        px = x + np.cos(angles)[:, None] * dists[None, :]
        py = y + np.sin(angles)[:, None] * dists[None, :]

        dx = px[:, :, None] - self.obstacles[:, 0]
        dy = py[:, :, None] - self.obstacles[:, 1]
        in_obstacle = (dx**2 + dy**2 <= self.obstacles[:, 2]**2).any(axis=2)
        out_of_map = (px < 0) | (px > self.map_size) | (py < 0) | (py > self.map_size)
        hit = in_obstacle | out_of_map

        ranges = np.full(self.n_rays, self.max_range)
        has_hit = hit.any(axis=1)
        ranges[has_hit] = dists[hit.argmax(axis=1)[has_hit]]   # first hit on each ray
        return ranges

    def _gps_position(self):
        noise = 0.0
        if self.gps_noise > 0:
            noise = self.rng.normal(0, self.gps_noise, size=2)
        return self.pos + self.offset_vec + noise

    def _get_obs(self):
        gx, gy = self._gps_position()
        dx, dy = self.goal[0] - gx, self.goal[1] - gy
        dist = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - self.yaw   # goal direction seen from the uav

        obs = np.concatenate([
            [dist / self.diag,
             np.sin(bearing), np.cos(bearing),
             self.v / self.v_max,
             self.w / self.w_max,
             gx / self.map_size * 2 - 1,
             gy / self.map_size * 2 - 1],
            self._lidar() / self.max_range,
        ])
        return np.clip(obs, -1, 1).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        start, self.goal = self._make_map()
        self.pos = start.copy()
        self.yaw = self.rng.uniform(-np.pi, np.pi)
        self.v, self.w = 0.0, 0.0
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)

        angle = self.rng.uniform(-np.pi, np.pi)
        self.offset_vec = self.gps_offset * np.array([np.cos(angle), np.sin(angle)])

        self.path = [self.pos.copy()]
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.v = (a[0] + 1) / 2 * self.v_max   # speed can't be negative
        self.w = a[1] * self.w_max

        self.yaw = (self.yaw + self.w * self.dt + np.pi) % (2 * np.pi) - np.pi
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw)])
        self.pos = self.pos + self.v * self.dt * heading
        self.steps += 1
        self.path.append(self.pos.copy())

        dist = np.linalg.norm(self.goal - self.pos)
        clear = self._clearance_with_walls(self.pos)

        reward = 10.0 * (self.prev_dist - dist)   # progress toward the goal
        reward -= 0.05                            # time penalty
        reward -= 0.05 * abs(a[1])                # avoid jerky turns
        if clear < 2.0:
            reward -= 0.5 * (2.0 - clear)         # too close to an obstacle
        self.prev_dist = dist

        terminated, truncated, event = False, False, "running"
        if clear < self.uav_radius:
            reward -= 100.0
            terminated, event = True, "collision"
        elif dist < self.goal_radius:
            reward += 100.0
            terminated, event = True, "goal"
        elif self.steps >= self.max_steps:
            truncated, event = True, "timeout"

        return self._get_obs(), float(reward), terminated, truncated, {"event": event}