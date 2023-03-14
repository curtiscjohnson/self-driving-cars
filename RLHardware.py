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

def preprocess_image(BGRimg):

    HSVimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HSV)
    HSVimg = cv2.bilateralFilter(HSVimg,9,75,75)
    # HSVimg = cv2.GaussianBlur(HSVimg, (5,5),0)
    blackImg = np.zeros(HSVimg.shape, dtype = np.uint8)
   
    # make true yellow
    lower_yellow = np.array([15,89,124])
    upper_yellow = np.array([100,255,255])
    mask=cv2.inRange(HSVimg,lower_yellow,upper_yellow)
    blackImg[mask>0] = (0,255,255)

    # make true red
    lower_red = np.array([0,50,146])
    upper_red = np.array([5,255,255])
    mask=cv2.inRange(HSVimg,lower_red,upper_red)
    blackImg[mask>0] = (0,0,255)

    # black out top 1/3 of image
    height, width, depth = blackImg.shape
    blackImg[0:height * 2 // 5,:,:] = (0, 0, 0)

    #black out bottom strip of image
    blackImg[height - 10:height, :, :] = (0, 0, 0)

    return blackImg

def setup_loading_model(action_space):
    N_CHANNELS = 3
    (HEIGHT, WIDTH) = (64, 64)
    observation_space = spaces.Box(
        low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
    )
    # model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")
    model = NatureCNN(observation_space, action_space, normalized_image=True)

    archive = zipfile.ZipFile("/home/car/Desktop/self-driving-cars/sb3_models/local/612/612_model_220000_steps.zip", 'r')
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
model_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/612/'
# with open(model_path+"config.txt", 'r') as f:
#   config = json.load(f)
# action_space = config['actions']
action_space = [-30, 0, 30]
model = setup_loading_model(action_space)

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
fastSpeed = 0.0
Car.drive(fastSpeed)

# driving loop
while True:
  
  Car.drive(fastSpeed)

  # get data from camera
  (time_, img, depth, accel, gyro) = rs.getData(False)

  # prepare image to go into network
  resizedImg = cv2.resize(img, (64, 64))
  preprocessedImg = preprocess_image(resizedImg)

  networkImg = np.moveaxis(preprocessedImg, 2, 0)

  # get steering angle
  action_idx = model(torch.from_numpy(networkImg/255).float()).max(0)[1].view(1,1)  
  angle = action_space[action_idx]

  # apply steering angle to car
  Car.steer(angle)

  # display what camera sees
  cv2.namedWindow("car", cv2.WINDOW_NORMAL)
  cv2.imshow("car", preprocessedImg)

  if (cv2.waitKey(1) == ord('q')):
      cv2.destroyAllWindows()
      break