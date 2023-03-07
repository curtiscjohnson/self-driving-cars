from stable_baselines3.common.monitor import Monitor
from CustomDuckieTownEnv import CustomDuckieTownSim
from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch import tensor
import wandb
from wandb.integration.sb3 import WandbCallback
from multiprocessing import Manager, Process
from stable_baselines3.common.vec_env import VecFrameStack



def make_env(display, config):
    env = CustomDuckieTownSim(
        config["camera_settings"],
        config["map_parameters"],
        config["car_parameters"],
        config["actions"],
        display,
    )
    env = Monitor(env)  # record stats such as returns
    return env


def train(config):


    run = wandb.init(
        project="self_driving_cars",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    env = make_env(False, config)

    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)

    # env = DummyVecEnv([make_env])

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",For some reason I can't get on github, but I'm pretty sure it's in the tile generator file. Definitely in the simulation folder, one of the tile classes


    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200
    # )

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
        # policy_kwargs=config["policy_kwargs"],
        tensorboard_log=f"runs/{run.id}",
        verbose=1,
        # seed=config["seed"]
    )

    trained_model = model.learn(
        total_timesteps=config["n_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=0,
            model_save_freq=1000,
            log='all',
        ),
    )
    run.finish()


if __name__ == "__main__":
    img_size = (64, 64)
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

#taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    config = {
        "n_timesteps": 1e6,  # sb3 dqn runs go up to 1e7 at most
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
    }


    # manager = Manager()
    # model_dict = manager.dict()
    # jobs = []
    # numWorkers = 1

    # for i in range(numWorkers):
    #     p = Process(target=train, args=(random.randint(0,100), model_dict))
    #     jobs.append(p)
    #     p.start()

    # for proc in jobs:
    #     proc.join()

    # print(model_dict.keys())

    train(config)
