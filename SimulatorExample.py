import random

from simulation import Simulator
import cv2

from simulation import coordinate
import numpy as np

from simple_pid import PID

import matplotlib.pyplot as plt

import lightning_mcqueen as lm
import time

# Can pass "cameraSettings" - see the camera.py file for options.
# M and D matrices not currently used, D could be used for distortion but that would need to be implemented in camera.py - wouldn't be too bad to do
# Units are pixels for resolution, degrees for fov, degrees for angle, and pixels for height.
cameraSettings = {
    "resolution": (1920, 1080),
    "fov": {"diagonal": 77}, # realsense diagonal fov is 77 degrees IIRC
    "angle": {"roll": 0, "pitch": 15, "yaw": 0}, # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math gets unstable
    # "angle": {"roll": 13, "pitch": 30, "yaw": 30}, # don't go too crazy with these, my code should be good up to like... 45 degrees probably? But the math gets unstable
    "height": 66 # 8 pixels/inch - represents how high up the camera is relative to the road
}

sim = Simulator(cameraSettings=cameraSettings)
realsense = sim.RealSense
Arduino = sim.Arduino
maxSteer = 30
maxSpeed = 3.0
speedUnits = 100
def updateSteer(val):
    global controlSteer, realSteer, Arduino
    controlSteer = val
    realSteer = controlSteer - maxSteer
    Arduino.setSteering(realSteer)

def updateSpeed(val):
    global controlSpeed, realSpeed, Arduino
    controlSpeed = val
    tempSpeed = controlSpeed - speedUnits/2
    realSpeed = (maxSpeed / (speedUnits / 2)) * tempSpeed
    Arduino.setSpeed(realSpeed)

controlSpeed = round(speedUnits/2)
controlSteer = maxSteer
realSpeed = updateSpeed(controlSpeed)
realSteer = updateSteer(controlSteer)

cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.namedWindow("car", cv2.WINDOW_NORMAL)
cv2.createTrackbar("speed", "map", controlSpeed, speedUnits, updateSpeed)
cv2.createTrackbar("steer", "map", controlSteer, maxSteer * 2, updateSteer)

# Can pass map parameters:
mapParameters = {
    "loops": 1,
    "size": (6, 6),
    "expansions": 5,
    "complications": 4
}

# Can also pass car parameters for max/min speed, etc
carParameters = {
    "wheelbase": 6.5, # inches, influences how quickly the steering will turn the car.  Larger = slower
    "maxSteering": 30.0, # degrees, extreme (+ and -) values of steering
    "steeringOffset": 0.0, # degrees, since the car is rarely perfectly aligned
    "minVelocity": 0.0, # pixels/second, slower than this doesn't move at all.
    "maxVelocity": 480.0, # pixels/second, 8 pixels/inch, so if the car can move 5 fps that gives us 480 pixels/s top speed
}

# sim.start(,carParameters=carParameters)
seed = 1333
random.seed(seed)
# random seed for consistent maps
# can also pass a start location if you know the code: (y tile index, x tile index, position index, direction index)
# - position index is from 0-(number of connections the tile has - 1), so a straight is 0 or 1, a t is 0, 1, or 2.
# - direction index is 0 or 1 for normal or reversed.
sim.start(mapSeed='real', mapParameters=mapParameters, carParameters=carParameters, startPoint=(0,4,0,0))

car = sim.ackermann

# I get ~20ms per loop on my computer, so roughly 1.5x real-world speed at 30fps.

## SETUP PID Controller
pid = PID()
pid.Ki = -.01*0
pid.Kd = -.01*0
pid.Kp = -30/350 #degrees per pixel
frameUpdate = 1
pid.sample_time = frameUpdate/30.0
pid.output_limits = (-30,30)
desXCoord = cameraSettings['resolution'][0]//2
pid.setpoint = desXCoord

i = 1
angle = 0
FAST_SPEED = 1.5
draw_bool = True
centers = []

turning = False

Arduino.setSpeed(FAST_SPEED)

while(True):
    img = realsense.getFrame() #time step entire simulation
    carPlace = car.getCoord()

    # control loop
    if i%frameUpdate == 0:
        i = 0
        centers = lm.get_yellow_centers(img)
        possible_turns = lm.identify_possible_turns(img.shape, centers)

        if len(possible_turns) > 0 and not turning:
            turn = lm.pick_turn(possible_turns)
            print(f"turning: {turn}")
            # set angle
            if turn == "right":
                angle = 20
            elif turn == "left":
                angle = -20
            else:
                angle = 0

            Arduino.setSteering(angle)
            turning = True
        elif len(possible_turns) == 0:
            blobToFollowCoords = centers[-1]
            blobX = blobToFollowCoords[0]

            angle = pid(blobX)
            # print(f"angle: {angle}")
            Arduino.setSteering(angle)
            Arduino.setSpeed(FAST_SPEED) 
            turning = False

    i+=1

    # Display Code
    newImg = realsense.currentImg.copy()
    newImg = cv2.line(newImg, carPlace.asInt(), (carPlace + 100*coordinate.unitFromAngle(car.currentState.theta, isRadians = True)).asInt(), (0, 255, 0), 3)
    newImg = cv2.line(newImg, carPlace.asInt(), (carPlace + 100*coordinate.unitFromAngle(car.currentState.theta + car.currentState.delta + car.currentState.steeringOffset, isRadians = True)).asInt(), (0, 0, 255), 3)
    # cv2.imshow("map", realsense.currentImg)
    cv2.imshow("map", newImg)
    if draw_bool:
        lm.draw_centers(img, centers)

        LEFT_X_THRESH = img.shape[1] // 4
        RIGHT_X_THRESH = int(img.shape[1] *3/4)
        Y_UPPER_THRESH = int(img.shape[0] *4.5/5)
        Y_LOWER_THRESH = int(img.shape[0] *3/5)

        # horizontal band
        img = cv2.line(img, (0, Y_LOWER_THRESH), (img.shape[1],Y_LOWER_THRESH), (0,255,0), thickness=5)
        img = cv2.line(img, (0, Y_UPPER_THRESH), (img.shape[1],Y_UPPER_THRESH), (0,255,0), thickness=5)

        # left and right
        img = cv2.line(img, (LEFT_X_THRESH, 0), (LEFT_X_THRESH,img.shape[0]), (0,255,0), thickness=5)
        img = cv2.line(img, (RIGHT_X_THRESH, 0), (RIGHT_X_THRESH,img.shape[0]), (0,255,0), thickness=5)

    cv2.imshow("car", img)
    statData = sim.getStats()

    if (cv2.waitKey(1) == ord('q')): # this simulator waits for a keypress every frame because otherwise it'd be really hard to control I think.
        # You could probably implement arrow key control, but... I didn't.  So.
        # Press q to quit, anything else to advance 1/30 second.
        break
    