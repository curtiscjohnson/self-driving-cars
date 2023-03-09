from Arduino import Arduino
from RealSense import *
import numpy as np
import cv2
import time as tm
import lightning_mcqueen as lm
import torch
import torch.nn as nn
from stable_baselines3 import DQN

def preprocess_image(BGRimg):

    HSVimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HSV)
    HSVimg = cv2.bilateralFilter(HSVimg,9,75,75)
    # HSVimg = cv2.GaussianBlur(HSVimg, (5,5),0)
    blackImg = np.zeros(HSVimg.shape, dtype = np.uint8)

    # make everything we don't care about black
    lower_thresh = np.array([0, 0, 203])
    upper_thresh = np.array([180, 255, 255])
    mask=cv2.inRange(HSVimg,lower_thresh,upper_thresh)
    BGRimg[mask==0]=(0,0,0)
   
    # make true yellow
    # 15, 100
    # 89, 255
    # 124, 255
    lower_yellow = np.array([15,89,124])
    upper_yellow = np.array([100,255,255])
    mask=cv2.inRange(HSVimg,lower_yellow,upper_yellow)
    blackImg[mask>0] = (0,255,255)

    # make true red
    lower_red = np.array([0,50,146])
    upper_red = np.array([5,255,255])
    mask=cv2.inRange(HSVimg,lower_red,upper_red)
    blackImg[mask>0] = (0,0,255)

    # # make true white
    # lower_white = np.array([26,0,160])
    # upper_white = np.array([152,60,255])
    # mask=cv2.inRange(HSVimg,lower_white,upper_white)
    # blackImg[mask>0] = (255,255,255)

    # black out top 1/3 of image
    height, width, depth = blackImg.shape
    blackImg[0:height * 2 // 5,:,:] = (0, 0, 0)

    #black out bottom strip of image
    blackImg[height - 10:height, :, :] = (0, 0, 0)

    return blackImg

def setup_loading_model():

    model = DQN.load("./sb3_models/local/650/650_model_760000_steps.zip")

# path = r'/fsg/hps22/self-driving-cars/testimg.jpg'
# preprocessedImg = preprocess_image(cv2.imread(path))
# cv2.imshow("car", preprocessedImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# load in RL model
model = setup_loading_model()
action_space = [-30, 0, 30]

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
fastSpeed = .8
Car.drive(fastSpeed)

# driving loop
while True:
  
  Car.drive(fastSpeed)

  # get data from camera
  (time_, img, depth, accel, gyro) = rs.getData(False)

  # prepare image to go into network
  preprocessedImg = preprocess_image(img)
  resizedImg = cv2.resize(preprocessedImg, (64, 64))

  # get steering angle
  action_idx, _ = model.predict(resizedImg, deterministic=True)    
  angle = action_space[action_idx]

  # apply steering angle to car
  Car.steer(angle)

  # display what camera sees
  cv2.namedWindow("car", cv2.WINDOW_NORMAL)
  cv2.imshow("car", resizedImg)

  if (cv2.waitKey(1) == ord('q')):
      cv2.destroyAllWindows()
      break