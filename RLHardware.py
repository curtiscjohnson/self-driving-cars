from Arduino import Arduino
from RealSense import *
import numpy as np
import cv2
import time as tm
import torch
import torch.nn as nn
from load_model_on_hardware_utils import setup_loading_model
from img_utils import preprocess_image
import argparse
from simple_pid import PID

from PID_Code.lightning_mcqueen import get_yellow_centers


parser = argparse.ArgumentParser()

parser.add_argument('--control', type=str, default='rl', help='rl or pid')
parser.add_argument('--yolo', type=bool, default=True, help='True: use yolo for sign recognition') 
parser.add_argument('--speed', type=float, default=0.8, help='.8 to 3.0')
parser.add_argument('--display', type=bool, default=True, help='True: show what camera is seeing')
opt = parser.parse_args()

if opt.control == 'rl':
    # load in RL model
    model_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016'
    zip_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016/curtis-20230325-124016_model_800000_steps.zip'
    model, config = setup_loading_model(model_path, zip_path)

elif opt.control == 'pid':
  ## SETUP PID Controller
  pid = PID()
  pid.Ki = -.01*0
  pid.Kd = -.01*0
  pid.Kp = -30/300 #degrees per pixel
  frameUpdate = 1
  pid.sample_time = frameUpdate/30.0
  pid.output_limits = (-30,30)
  desXCoord = 640//2 #!hard coded to 640x480 
  pid.setpoint = desXCoord
else:
  raise argparse.ArgumentTypeError("wrong contorl method")

# initialize car
Car = Arduino("/dev/ttyUSB0", 115200)  
Car.zero(1440)
Car.pid(True)

# initialize realsense camera
enableDepth = True
rs = RealSense("/dev/video2", RS_VGA)
writer = None

# start camera
rgb = rs.getData(False)

# start car
fastSpeed = opt.speed
Car.drive(fastSpeed)

  
# driving loop
while True:
  # get data from camera
  img = rs.getData(False)

  if opt.control == 'rl':
    # prepare image to go into network
    preprocessedImg = preprocess_image(
      img,
      removeBottomStrip=True, #should always do this on hardware
      blackAndWhite=config["blackAndWhite"],
      addYellowNoise=False, #no need to add noise to real world
      use3imgBuffer=config["use3imgBuffer"],
      yellow_features_only=config["yellow_features_only"]
      )
    
    networkImg = np.moveaxis(preprocessedImg, 2, 0)

    # get steering angle
    with torch.no_grad():
      action_idx = model(torch.from_numpy(networkImg/255).float().cuda()).max(0)[1].view(1,1)  
    angle = config["actions"][action_idx]
    
  elif opt.control == 'pid':
    # prepare image to go into network
    centers = get_yellow_centers(img)
    if centers != "None":
      angle = pid(centers[-1][0])


  Car.drive(fastSpeed)
  Car.steer(angle)

  # display what camera sees
  if opt.display:
    cv2.namedWindow("preprocessed", cv2.WINDOW_NORMAL)
    cv2.imshow("preprocessed", img)

    if (cv2.waitKey(1) == ord('q')):
        cv2.destroyAllWindows()
        break