from Arduino import Arduino
from RealSense import *
import numpy as np
import cv2
import time as tm
import torch
import torch.nn as nn
import zipfile
# from stable_baselines3 import DQN
from utils_network import NatureCNN
from gym import spaces
import json
from img_utils import preprocess_image


def setup_loading_model(config):
    N_CHANNELS = 3
    (WIDTH, HEIGHT) = config["camera_settings"]["resolution"]
    observation_space = spaces.Box(
        low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
    )
    # model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")
    model = NatureCNN(observation_space, config["actions"], normalized_image=True)

    archive = zipfile.ZipFile("/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016/curtis-20230325-124016_model_800000_steps.zip", 'r')
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

    return model

# path = r'/fsg/hps22/self-driving-cars/testimg.jpg'
# preprocessedImg = preprocess_image(cv2.imread(path))
# cv2.imshow("car", preprocessedImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# load in RL model
model_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016'
with open(model_path+"/config.txt", 'r') as f:
  config = json.load(f)
model = setup_loading_model(config)

print(config)

# initialize realsense camera
enableDepth = True
rs = RealSense("/dev/video2", RS_VGA, enableDepth)
writer = None

# initialize car
Car = Arduino("/dev/ttyUSB0", 115200)  
Car.zero(1440)
Car.pid(1)

# start camera
(time_, rgb, depth, accel, gyro) = rs.getData(False)

# start car
fastSpeed = 0.8
Car.drive(fastSpeed)

model.eval()
model.cuda()

# driving loop
while True:
  Car.drive(fastSpeed)

  # get data from camera
  (time_, img, depth, accel, gyro) = rs.getData(False)

  resizedImg = cv2.resize(img, tuple(config["camera_settings"]["resolution"]))

  # prepare image to go into network
  preprocessedImg = preprocess_image(
    resizedImg,
    removeBottomStrip=True, #should always do this on hardware
    blackAndWhite=config["blackAndWhite"],
    addYellowNoise=False, #no need to add noise to real world
    use3imgBuffer=config["use3imgBuffer"],
    yellow_features_only=config["yellow_features_only"]
    )

  # print(preprocessedImg.shape)

  # print(config["camera_settings"]["resolution"])
  # cv2.namedWindow("resizedImg", cv2.WINDOW_NORMAL)
  # cv2.imshow("resizedImg", resizedImg)
  networkImg = np.moveaxis(preprocessedImg, 2, 0)

  start = time.time()
  # get steering angle
  with torch.no_grad():
    action_idx = model(torch.from_numpy(networkImg/255).float().cuda()).max(0)[1].view(1,1)  
  angle = config["actions"][action_idx]
  # apply steering angle to car
  end = time.time()

  Car.steer(angle)
  print(f'Loop Time: {end-start}')

  # # display what camera sees
  cv2.namedWindow("preprocessed", cv2.WINDOW_NORMAL)
  cv2.imshow("preprocessed", preprocessedImg)

  if (cv2.waitKey(1) == ord('q')):
      cv2.destroyAllWindows()
      break