# python3 ECEnRacer.py
''' 
This program is for ECEN-631 BYU Race
*************** RealSense Package ***************
From the Realsense camera:
	RGB Data
	Depth Data
	Gyroscope Data
	Accelerometer Data
*************** Arduino Package ****************
	Steer(int degree) : -30 (left) to +30 (right) degrees
	Drive(float speed) : -3.0 to 3.0 meters/second
	Zero(int PWM) : Sets front wheels going straight around 1500
	Encoder() : Returns current encoder count.  Reset to zero when stop
	Pid(int flag) : 0 to disable PID control, 1 to enable PID control
	KP(float p) : Proporation control 0 ~ 1.0 : how fast to reach the desired speed.
	KD(float d) : How smoothly to reach the desired speed.

	EXTREMELY IMPORTANT: Read the user manual carefully before operate the car

	# If you get cannot get frame error: use 'sudo pkill -9 python3.6' and wait 15 seconds
**************************************
'''

# import the necessary packages
from Arduino import Arduino
from RealSense import *
import numpy as np
import imutils
import cv2
from simple_pid import PID
import time as tm
import lightning_mcqueen as lm
import detect_lane as dl

enableDepth = True
rs = RealSense("/dev/video2", RS_VGA, enableDepth)		# RS_VGA, RS_720P, or RS_1080P
writer = None

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac

Car.zero(1440)      # Set car to go straight.  Change this for your car.
Car.pid(1)          # Use PID control

(time_, rgb, depth, accel, gyro) = rs.getData(False)
# cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)

## SETUP PID Controller
pid = PID()
pid.Ki = -.01*0
pid.Kd = -.01*0
pid.Kp = -30/250 #degrees per pixel
frameUpdate = 1
pid.sample_time = frameUpdate/30.0
pid.output_limits = (-30,30)
desXCoord = rgb.shape[0]*3/5
pid.setpoint = desXCoord

i = 1
angle = 0
FAST_SPEED = 1.3
SLOW_SPEED = 0.5
speed = FAST_SPEED
blob_lost = False
draw_bool = True
centers = []


# # You can use kd and kp commands to change KP and KD values.  Default values are good.
# # loop over frames from Realsense
j = 0
while(True):
	Car.drive(FAST_SPEED)
	(time_, img, depth, accel, gyro) = rs.getData(False)

	# control loop
	if i % frameUpdate == 0:

		i = 0
		centers = lm.get_yellow_centers(img)

		if centers != "None":
			blobToFollowCoords = centers[-1]
			blobX = blobToFollowCoords[0]
			angle = pid(blobX)
			Car.steer(angle)

	# Display Yellow Centers
	if draw_bool:
		lm.draw_centers(img, centers)

	cv2.imshow("car", img)

	i+=1
	if (cv2.waitKey(1) == ord('q')):
		cv2.destroyAllWindows()
		break

