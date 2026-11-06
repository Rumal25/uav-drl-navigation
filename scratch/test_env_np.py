import sys
import numpy as np
from env_np import UAVEnv

sys.path.append("..")
from uav_env import UAVNavEnv   # old gymnasium version, used only for comparison


# 1. same seed should give the same map
a, b = UAVEnv(), UAVEnv()
a.reset(seed=7)
b.reset(seed=7)
print("same seed, same map:", np.allclose(a.obstacles, b.obstacles))


# 2. new env should behave exactly like the old one
new_env, old_env = UAVEnv(), UAVNavEnv()
all_same = True

for seed in [0, 1, 2]:
    obs_new, _ = new_env.reset(seed=seed)
    obs_old, _ = old_env.reset(seed=seed)
    same = np.allclose(obs_new, obs_old)

    actions = np.random.default_rng(100 + seed)
    done = False
    while not done:
        act = actions.uniform(-1, 1, size=2)
        obs_new, r_new, term_new, trunc_new, info = new_env.step(act)
        obs_old, r_old, term_old, trunc_old, _ = old_env.step(act)
        same = same and np.allclose(obs_new, obs_old) and np.isclose(r_new, r_old) \
            and term_new == term_old
        done = term_new or trunc_new

    print(f"seed {seed}: same as old env = {same}, "
          f"ended with {info['event']} after {new_env.steps} steps")
    all_same = all_same and same

print("new env matches old env:", all_same)


# 3. run some random episodes
env = UAVEnv(seed=0)
results = []
for ep in range(20):
    obs, _ = env.reset()
    assert obs.shape == (43,) and obs.dtype == np.float32
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.rng.uniform(-1, 1, size=2))
        done = term or trunc
    results.append(info["event"])

print("20 random episodes:", {k: results.count(k) for k in set(results)})