import json
from gym import spaces
import numpy as np
from utils_network import NatureCNN
import zipfile
import torch

def setup_loading_model(model_path:str, zip_path:str):

    with open(model_path+"/config.txt", 'r') as f:
        config = json.load(f)

    N_CHANNELS = 3
    (WIDTH, HEIGHT) = config["camera_settings"]["resolution"]
    observation_space = spaces.Box(
        low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
    )
    # model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")
    model = NatureCNN(observation_space, config["actions"], normalized_image=True)

    archive = zipfile.ZipFile(zip_path, 'r')
    path = archive.extract('policy.pth')
    state_dict = torch.load(path)
    # print('\nState Dict:', state_dict.keys(), '\n')

    new_state_dict = {}
    for old_key in state_dict.keys():
        if "q_net.q_net" in old_key:
          new_key = "action_output." + old_key.split(".")[-1]
        elif "q_net_target" not in old_key:
          new_key = ".".join(old_key.split(".")[2:])
  
        new_state_dict[new_key] = state_dict[old_key]
    
    # print('\nNew State Dict:', new_state_dict.keys(), '\n')
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()

    pretty = json.dumps(config, indent=4)
    print(f"Model Config\n{pretty}")
    return model, config

