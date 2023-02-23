from Arduino import Arduino
from RealSense import *
import numpy as np
import cv2
import time as tm
import lightning_mcqueen as lm
import torch
import torch.nn as nn

# Q-Value Network
class QNetwork(nn.Module):
  def __init__(self, action_size, in_channels=3, cnn_outchannels=1, hidden_size=128):
    super().__init__()
  
    self.cnn = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=cnn_outchannels*16, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=cnn_outchannels*16, out_channels=cnn_outchannels*32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=cnn_outchannels*32, out_channels=cnn_outchannels, kernel_size=3, stride=1, padding=1),
                            nn.ReLU()
                            )

    self.controller = nn.Sequential(
                            nn.Linear(128*72, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, action_size)
                            )

    
  def forward(self, img_batch):
    """Estimate q-values given image
    """
    cnn_output = self.cnn(img_batch.permute([0, 3, 1, 2]))
    return self.controller(cnn_output.view(cnn_output.size(0), -1)) # Have to resize the output of the cnn to be accepted  by the linear layer

# load in RL model
path = './rl_models/working_2actmodel.pt'
action_size = 2
action_space = [-30,30]
model = QNetwork(action_size)
model = torch.load(path)
model.eval()

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
  resizedImg = cv2.resize(img, (72, 128))
  frame = torch.tensor(resizedImg, dtype=torch.float32).unsqueeze(0).cuda()

  # get steering angle
  with torch.no_grad():
    action_idx = model(frame).max(1)[1].view(1, 1)    
  angle = action_space[action_idx]

  # apply steering angle to car
  Car.steer(angle)

  # display what camera sees
  cv2.imshow("car", img)

  if (cv2.waitKey(1) == ord('q')):
      cv2.destroyAllWindows()
      break