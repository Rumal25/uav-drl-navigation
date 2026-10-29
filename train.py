import argparse
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from uav_env import UAVNavEnv


class SuccessCallback(BaseCallback):
    """After each episode, record goal/collision; log the average over the last 100 episodes."""
    def __init__(self):
        super().__init__()
        self.results = []          # 1 = goal, 0 = not goal
        self.crashes = []

    def _on_step(self):
        for info in self.locals["infos"]:          # one info per parallel environment
            if "episode" in info:                  # an episode just ended
                self.results.append(info["event"] == "goal")
                self.crashes.append(info["event"] == "collision")
        if self.results:
            self.logger.record("rollout/success_rate", np.mean(self.results[-100:]))
            self.logger.record("rollout/collision_rate", np.mean(self.crashes[-100:]))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_name = f"{args.algo}_seed{args.seed}"

    run = wandb.init(project="uav-drl", name=run_name, config=vars(args), sync_tensorboard=True)

    if args.algo == "PPO":
        env = make_vec_env(UAVNavEnv, n_envs=4, seed=args.seed)   # 4 parallel copies
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=1024,          # steps per env before each update (x4 envs = 4096)
            batch_size=256,        # samples per gradient step
            n_epochs=10,           # how many times to learn from the same data
            gamma=0.99,            # discount: how much future reward matters
            gae_lambda=0.95,       # smooths the advantage estimate
            clip_range=0.2,        # limits how much the policy changes per update
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=f"runs/{run.id}", device="cpu", seed=args.seed, verbose=1,
        )
    else:
        env = make_vec_env(UAVNavEnv, n_envs=1, seed=args.seed)
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=300_000,   # replay buffer: how many past steps to remember
            batch_size=256,
            learning_starts=5_000, # fly randomly first to fill the buffer
            gamma=0.99,
            tau=0.005,             # how slowly the target network follows
            ent_coef="auto",       # SAC tunes its own exploration
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=f"runs/{run.id}", device="cpu", seed=args.seed, verbose=1,
        )
    checkpoint = CheckpointCallback(save_freq=25_000, save_path=f"models/{run_name}_ckpt",
                                name_prefix=run_name)
    model.learn(total_timesteps=args.steps,
            callback=[SuccessCallback(), WandbCallback(), checkpoint])
    model.save(f"models/{run_name}")
    print("saved:", f"models/{run_name}.zip")
    run.finish()


if __name__ == "__main__":      # needed on Windows
    main()