from stable_baselines3.common.monitor import Monitor
from CustomDuckieTownEnvMLP import CustomDuckieTownSim
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import json
import time



def make_env(display, config):
    env = CustomDuckieTownSim(
        config["camera_settings"],
        config["map_parameters"],
        config["car_parameters"],
        config["num_blobs"],
        config["actions"],
        config["speed"],
        config["randomizeCameraParamsOnReset"],
        display,
    )
    env = Monitor(env)  # record stats such as returns
    return env


def train(config, display):

    env = make_env(display, config)
    run = time.strftime("%Y%m%d-%H%M%S")
    netid = "dcheney1"
    path_to_jdrive = f"/fsg/{netid}/groups/self-driving"
    print(f'Run_id: {run}')
    
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
            save_freq=1e5,
            save_path=model_save_path,
            name_prefix=f"{run}_model",
        ),
    )

    # open/create file for writing config
    with open(model_save_path+'config.txt', 'w') as convert_file:
        convert_file.write(json.dumps(config))

if __name__ == "__main__":
    shrinkFactor = 1
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
        "n_timesteps": 2e6,  # sb3 dqn runs go up to 1e7 at most
        "num_blobs":10,
        "notes":"MLP with yellow blobs. Trying more blobs",


        "policy": "MlpPolicy",
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
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.01,
        "randomizeCameraParamsOnReset":True,
        "speed": 1.0,
    }

    train(config, False)

