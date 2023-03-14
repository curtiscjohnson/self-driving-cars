import json

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from torch import tensor

from CustomDuckieTownEnv import CustomDuckieTownSim


def make_env(display, config):
    env = CustomDuckieTownSim(
        config["camera_settings"],
        config["map_parameters"],
        config["car_parameters"],
        config["actions"],
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

def validate(model, config):
    env = make_env(True, config)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()



if __name__ == "__main__":

    model_run_id = 8818761
    model_path = f"./sb3_models/local/{model_run_id}/"
    # model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")
    # model = DQN.load("./sb3_models/local/91/91_model_1000000_steps.zip")
    # model = DQN.load("./sb3_models/local/650/650_model_1000000_steps.zip")
    model = DQN.load(model_path+f"{model_run_id}_model_10_steps.zip")

    with open(model_path+"config.txt", 'r') as f:
        config = json.load(f)
    
    # print(config)

    validate(model, config)

