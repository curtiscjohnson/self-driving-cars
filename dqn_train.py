from stable_baselines3.common.monitor import Monitor
from CustomDuckieTownEnv import CustomDuckieTownSim
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from datetime import datetime
from multiprocessing import Manager, Process
from stable_baselines3.common.vec_env import VecFrameStack
import random
import json



def make_env(display, config):
    env = CustomDuckieTownSim(
        config["camera_settings"],
        config["map_parameters"],
        config["car_parameters"],
        config["actions"],
        config["max_episode_length"],
        display,
    )
    env = Monitor(env)  # record stats such as returns

    # Frame-stacking with 4 frames
    # env = VecFrameStack(env, n_stack=4)

    # env = DummyVecEnv([make_env])

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",For some reason I can't get on github, but I'm pretty sure it's in the tile generator file. Definitely in the simulation folder, one of the tile classes

    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200
    #
    return env


def train(config, sync2wandb=False):

    env = make_env(False, config)
    if sync2wandb:
        run = wandb.init(
            project="sb3",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

        model = DQN(
            config["policy"],
            env,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            gamma=config["gamma"],
            target_update_interval=config["target_update_interval"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            tensorboard_log=f"sb3_runs/wandb/{run.id}",
            verbose=1,
        )

        final_model = model.learn(
            total_timesteps=config["n_timesteps"],
            tb_log_name=model.tensorboard_log,
            callback=WandbCallback(
                # gradient_save_freq=100,
                model_save_path=f"sb3_models/wandb/{run.id}",
                verbose=1,
                model_save_freq=10000,
                log="all",
            ),
            # https://github.com/wandb/wandb/blob/72eeaa2c975cddd540a72223fa11c3f2537371a6/wandb/integration/sb3/sb3.py
            # ! I think wandb overwrites model.zip every time...
        )

        run.finish()

        
    else:
        # Get the run number as a timestamp (subtracting to remove a few digits)
        run = int(datetime.timestamp(datetime.now()) - 1670000000)

        model = DQN(
            config["policy"],
            env,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            gamma=config["gamma"],
            target_update_interval=config["target_update_interval"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            tensorboard_log=f"./sb3_runs/local/{run}/",
            verbose=1,
        )
        model_save_path = f"./sb3_models/local/{run}/"
        final_model = model.learn(
            total_timesteps=config["n_timesteps"],
            tb_log_name=model.tensorboard_log,
            callback=CheckpointCallback(
                save_freq=1e5,
                save_path=model_save_path,
                name_prefix=f"{run}_model",
            ),
        )

        # open/create file for writing config
        with open(model_save_path+'config.txt', 'w') as convert_file:
            convert_file.write(json.dumps(config))

if __name__ == "__main__":
    img_size = (128, 72)
    cameraSettings = {
        # "resolution": (1920, 1080),
        "resolution": img_size,
        "fov": {"diagonal": 77},  # realsense diagonal fov is 77 degrees IIRC
        "angle": {
            "roll": 0,
            "pitch": 0,
            "yaw": 0,
        },  # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math gets unstable
        # "angle": {"roll": 13, "pitch": 30, "yaw": 30}, # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math gets unstable
        "height": 66,  # 8 pixels/inch - represents how high up the camera is relative to the road
    }

    mapParameters = {"loops": 1, "size": (6, 6), "expansions": 5, "complications": 4}

    # Can also pass car parameters for max/min speed, etc
    carParameters = {
        "wheelbase": 6.5,  # inches, influences how quickly the steering will turn the car.  Larger = slower
        "maxSteering": 30.0,  # degrees, extreme (+ and -) values of steering
        "steeringOffset": 0.0,  # degrees, since the car is rarely perfectly aligned
        "minVelocity": 0.0,  # pixels/second, slower than this doesn't move at all.
        "maxVelocity": 480.0,  # pixels/second, 8 pixels/inch, so if the car can move 5 fps that gives us 480 pixels/s top speed
    }

    # taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    config = {
        "n_timesteps": 5e6,  # sb3 dqn runs go up to 1e7 at most
        "policy": "CnnPolicy",
        "env": "CustomDuckieTown",
        "actions": [-30, 0, 30],
        "camera_settings": cameraSettings,
        "map_parameters": mapParameters,
        "car_parameters": carParameters,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "buffer_size": 100000,
        "learning_starts": 100000,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "max_episode_length":1000,
        "yellow_image_noise":True,
        "notes":"add notes here"
    }

    train(config, False)

