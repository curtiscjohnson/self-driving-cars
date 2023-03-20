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
import time



def make_env(display, config):
    env = CustomDuckieTownSim(
        config["camera_settings"],
        config["map_parameters"],
        config["car_parameters"],
        config["actions"],
        config["max_episode_length"],
        config["yellow_image_noise"],
        config["blackAndWhite"],
        config["use3imgBuffer"],
        config["randomizeCameraParamsOnReset"],
        display,
    )
    env = Monitor(env)  # record stats such as returns
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
            verbose=0,
        )

        final_model = model.learn(
            total_timesteps=config["n_timesteps"],
            tb_log_name=model.tensorboard_log,
            callback=WandbCallback(
                # gradient_save_freq=100,
                model_save_path=f"sb3_models/wandb/{run.id}",
                verbose=0,
                model_save_freq=10000,
                log="all",
            ),
            # https://github.com/wandb/wandb/blob/72eeaa2c975cddd540a72223fa11c3f2537371a6/wandb/integration/sb3/sb3.py
            # ! I think wandb overwrites model.zip every time...
        )

        run.finish()

        
    else:

        run = time.strftime("%Y%m%d-%H%M%S")
        netid = "cjohns94"
        path_to_jdrive = f"/fsg/{netid}/groups/self-driving"
        
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
            tensorboard_log=f"{path_to_jdrive}/sb3_runs/{run}",
            verbose=0,
        )
            
        model_save_path = f"{path_to_jdrive}/sb3_models/{run}/"
        final_model = model.learn(
            total_timesteps=config["n_timesteps"],
            # tb_log_name=f"{run}",
            progress_bar=True,
            callback=CheckpointCallback(
                save_freq=20,
                save_path=model_save_path,
                name_prefix=f"{run}_model",
            ),
        )

        # open/create file for writing config
        with open(model_save_path+'config.txt', 'w') as convert_file:
            convert_file.write(json.dumps(config))

if __name__ == "__main__":
    shrinkFactor = 10
    img_size = (1920//shrinkFactor,1080//shrinkFactor) #! must be (cols, rows) i.e. (width, height)
    cameraSettings = {
        # "resolution": (1920, 1080),
        "resolution": img_size,
        "fov": {"diagonal": 77},  # realsense diagonal fov is 77 degrees IIRC
        "angle": {
            "roll": 0,
            "pitch": -5,
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
        "n_timesteps": 60,  # sb3 dqn runs go up to 1e7 at most
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
        "max_episode_length":1800, #60 seconds of driving without crashing, hopefully not memorize one loop so much.
        "yellow_image_noise":False,
        "blackAndWhite": True,
        "use3imgBuffer":True, #! only works if blackAndWhite is true
        "randomizeCameraParamsOnReset":True,
        "notes":"On auto-19"
    }

    train(config, False)

