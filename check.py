from gymnasium.utils.env_checker import check_env
from uav_env import UAVNavEnv

env = UAVNavEnv()
check_env(env)                       # Gymnasium's official test: sizes, types, seeding
print("check_env passed!")

obs, info = env.reset(seed=0)
print("first obs shape:", obs.shape)
for ep in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    print(f"episode {ep}: {info['event']} after {env.steps} steps")
