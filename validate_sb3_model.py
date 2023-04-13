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
        action_angles=config["actions"],
        # addYellowNoise=config["yellow_image_noise"],
        blackAndWhite=config["blackAndWhite"],
        use3imgBuffer=config["use3imgBuffer"],
        randomizeCameraParamsOnReset=config["randomizeCameraParamsOnReset"],
        # config["yellow_features_only"],
        display=display,
    )
    env = Monitor(env)  # record stats such as returns
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

    model_run_id = 650
    steps = 760000
    model_path = f"/fsg/cjohns94/groups/self-driving/sb3_models/cjohns94-20230411-132349/"
    # model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")
    # model = DQN.load("./sb3_models/local/91/91_model_1000000_steps.zip")
    # model = DQN.load("./sb3_models/local/650/650_model_1000000_steps.zip")
    model = DQN.load(model_path+"cjohns94-20230411-132349_model_4450000_steps")

    with open(model_path+"/config.txt", 'r') as f:
        config = json.load(f)
    
    # print(config)

    validate(model, config)
