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

enableDepth = True
rs = RealSense("/dev/video2", RS_VGA, enableDepth)		# RS_VGA, RS_720P, or RS_1080P
writer = None

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)                # Linux
#Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac

Car.zero(1440)      # Set car to go straight.  Change this for your car.
Car.pid(1)          # Use PID control

(time, rgb, depth, accel, gyro) = rs.getData(False)
cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)

################################ FUNCTIONS ################################

def get_yellow_blob_x(bgr_img):
    """gets x coord of centerline blob to follow

    Args:
        img (_type_): BGR image from realsense camera

    Returns:
        float: x coordinate of the closest blob
    """
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv_img)
    # plt.show()
    # Get grayscale image with only centerline (yellow colors)
    lower_yellow = np.array([21,126,191])
    upper_yellow = np.array([75,255,255])
    centerline_gray_img = cv2.inRange(hsv_img, lower_yellow, upper_yellow) # get only yellow colors in image

    # Get Contours for center line blobs
    contours, hierarchy = cv2.findContours(centerline_gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy > hsv_img.shape[0]//2:
                centers.append((cx, cy))

    centers.sort(key = lambda x: x[1])

    # Make sure that their is a yellow blob found
    if len(centers) == 0:
        return "None"
    else:
        return centers

def draw_centers(img, centers):
    # Draws given centers onto given image
    if len(centers) > 1 and centers != "None":
        # print(f"centers;: {centers}")
        for point in centers:
            cv2.circle(img, point, 7, (0, 0, 255), -1) 
            # args: img to draw on, point to draw, size of circle, color, line width (-1 defaults to fill)

################################ MAIN LOOP ################################

## SETUP PID Controller
pid = PID()
pid.Ki = -.01*0
pid.Kd = -.01*0
pid.Kp = -30/350 #degrees per pixel
frameUpdate = 1
pid.sample_time = frameUpdate/30.0
pid.output_limits = (-30,30)
desXCoord = rgb.shape[0]//3
pid.setpoint = desXCoord

i = 1
angle = 0
FAST_SPEED = .8
SLOW_SPEED = 0.5
speed = FAST_SPEED
blob_lost = False
draw_centers_bool = True
centers = []

Car.drive(1.3)
tm.sleep(.1)

# You can use kd and kp commands to change KP and KD values.  Default values are good.
# loop over frames from Realsense
while(True):
	(time, rgb, depth, accel, gyro) = rs.getData(False)

	# control loop
	if i%frameUpdate == 0:
		i = 0
		centers = get_yellow_blob_x(rgb)
		# if len(centers) > 4:
		#     blobToFollowCoords = centers[-5]
		# else:

		if centers == "None" and not blob_lost:
			blob_lost = True
			speed = SLOW_SPEED
			angle = -30.0 
		elif centers != "None" or not blob_lost:
			blobToFollowCoords = centers[-1]
			blobX = blobToFollowCoords[0]

			blob_lost = False
			speed = FAST_SPEED
			angle = pid(blobX)

		Car.steer(angle)
		Car.drive(speed)
	i+=1

	if draw_centers_bool:
		draw_centers(rgb, centers)
	
	cv2.imshow("RGB", rgb)

	if (cv2.waitKey(1) == ord('q')):
		cv2.destroyAllWindows()
		break


#     cv2.imshow("Depth", depth)

#     '''
#     Add your code to process rgb, depth, IMU data
#     '''

#     '''
#     Control the Car
#     '''
#     car.Steer

#     '''
#    	IMPORTANT: Never go full speed. Use CarTest.py to selest the best speed for you.
#     Car can switch between positve and negative speed (go reverse) without any problem.
#     '''
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
# del rs
# del Car

