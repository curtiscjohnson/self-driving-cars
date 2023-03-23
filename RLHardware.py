from Arduino import Arduino
from RealSense import *
import numpy as np
import cv2
import time
import torch
import zipfile
from utils_network import NatureCNN
from gym import spaces
import json
from img_utils import preprocess_image
import matplotlib.pyplot as plt

def setup_loading_model(zip_path, config):
    N_CHANNELS = 3
    (WIDTH, HEIGHT) = config["camera_settings"]["resolution"]
    observation_space = spaces.Box(
        low=0, high=1, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
    )
    model = NatureCNN(observation_space, config["actions"], normalized_image=True)

    archive = zipfile.ZipFile("/home/car/Desktop/self-driving-cars/sb3_models/local/8826224/8826224_model_4600000_steps.zip", 'r')
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


def profiling_loop(model, rs, config, Car, speed, drive=True):
  j = 0
  history = []

  while j < 300:
    if drive:
      Car.drive(speed)

    start1 = time.time()
    img = rs.getData(False)
    getdata = time.time() - start1

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # prepare image to go into network
    start = time.time()
    resizedImg = cv2.resize(img, tuple(config["camera_settings"]["resolution"]))
    resize = time.time() - start

    start = time.time()
    preprocessedImg = preprocess_image(
      resizedImg,
      removeBottomStrip=True, #should always do this on hardware
      blackAndWhite=False, #config["blackAndWhite"],
      addYellowNoise=False, #no need to add noise to real world
      use3imgBuffer=False#config["use3imgBuffer"]
      )
    process = time.time() - start

    start = time.time()
    networkImg = np.moveaxis(preprocessedImg, 2, 0)
    moveaxis = time.time() - start

    # get steering angle
    start = time.time()
    action_idx = model(torch.from_numpy(networkImg/255).float()).max(0)[1].view(1,1)  
    getaction = time.time() - start
    end = time.time()

    angle = config["actions"][action_idx]
    if drive:
      Car.steer(angle)
    history.append(end-start1)
    print(f'getdata: {getdata}, resize: {resize}, process: {process}, moveaxis: {moveaxis}, getaction: {getaction}, loop time: {end-start1}')
    j += 1
  plt.hist(history, bins=50)
  plt.xlim((0.00, 0.4))
  plt.show()

def driving_loop(model, rs, config, Car, speed, drive=True):
  while True:
    if drive:
      Car.drive(speed)

    img = rs.getData(False)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # prepare image to go into network
    resizedImg = cv2.resize(img, tuple(config["camera_settings"]["resolution"]))

    preprocessedImg = preprocess_image(
      resizedImg,
      removeBottomStrip=True, #should always do this on hardware
      blackAndWhite=True,#,config["blackAndWhite"],
      addYellowNoise=True, #no need to add noise to real world
      use3imgBuffer=False#config["use3imgBuffer"]
      )

    networkImg = np.moveaxis(preprocessedImg, 2, 0)

    # get steering angle
    action_idx = model(torch.from_numpy(networkImg/255).float()).max(0)[1].view(1,1)  

    angle = config["actions"][action_idx]
    if drive:
      Car.steer(angle)

    # # display what camera sees
    # cv2.namedWindow("preprocessed", cv2.WINDOW_NORMAL)
    # cv2.imshow("preprocessed", preprocessedImg)

    # if (cv2.waitKey(1) == ord('q')):
    #     cv2.destroyAllWindows()
    #     break



if __name__=='__main__':
  # load in RL model
  model_name = 8826224
  steps = 4600000
  model_path = f'/home/car/Desktop/self-driving-cars/sb3_models/local/{model_name}/'
  zip_path = model_path + f'{model_name}_{steps}_steps.zip'
  time_profiling = True

  speed = .8
  drive = False

  with open(model_path+"config.txt", 'r') as f:
    config = json.load(f)
  model = setup_loading_model(zip_path, config)
  print(config)

  ## initialize realsense camera
  rs = RealSense("/dev/video2", RS_VGA)

  ## start camera
  img = rs.getData(False)

  ## start car
  if drive:
    # initialize car
    Car = Arduino("/dev/ttyUSB0", 115200)  
    Car.zero(1440)
    Car.pid(1)
    Car.drive(speed)
  else:
    Car = None

if not time_profiling:
  driving_loop(model, rs, config, Car, speed, drive=drive)
else:
  profiling_loop(model, rs, config, Car, speed, drive=drive)