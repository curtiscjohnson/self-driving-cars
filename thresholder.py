from __future__ import print_function
from tkinter.ttk import Frame
import cv2 as cv
import argparse

max_value = 255
max_value_H = 360//2
low_H = 0
low_L = 0
low_S = 0
high_H = max_value_H
high_L = max_value
high_S = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_L_name = 'Low L'
low_S_name = 'Low S'
high_H_name = 'High H'
high_L_name = 'High L'
high_S_name = 'High S'

## [low]
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
## [low]

## [high]
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
## [high]

def on_low_L_thresh_trackbar(val):
    global low_L
    global high_L
    low_L = val
    low_L = min(high_L-1, low_L)
    cv.setTrackbarPos(low_L_name, window_detection_name, low_L)

def on_high_L_thresh_trackbar(val):
    global low_L
    global high_L
    high_L = val
    high_L = max(high_L, low_L+1)
    cv.setTrackbarPos(high_L_name, window_detection_name, high_L)

def on_low_V_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

## [cap]
cap = cv.VideoCapture("/dev/video2")
## [cap]

## [window]
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
## [window]

## [trackbar]
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_L_name, window_detection_name , low_L, max_value, on_low_L_thresh_trackbar)
cv.createTrackbar(high_L_name, window_detection_name , high_L, max_value, on_high_L_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
## [trackbar]

while True:
    ## [while]
    ret, frame = cap.read()
    # frame = cv.imread('./sim_testing_images/64by64sim_image.png')
    if frame is None:
        break

    BGRimg = cv.bilateralFilter(
        frame, 3, 50, 50
    )  # theoretically good at removing noise but keeping sharp lines.

    # black out top 1/3 of image
    height, width, depth = BGRimg.shape
    BGRimg[0 : height // 2, :, :] = (0, 0, 0)

    # black out bottom strip of image
    BGRimg[height - 25 : height, :, :] = (0, 0, 0)

    # HLS COLOR SPACE THRESHOLDS
    # Yellow : 18-57 H, 73-216 L, 85-255 S
    # Red: 0-15 H, 101-255 L, 85-255 S
    # White: Pretty hard
    
    frame_HLS = cv.cvtColor(BGRimg, cv.COLOR_BGR2HLS)
    frame_gray = cv. cvtColor(BGRimg, cv.COLOR_BGR2GRAY)

    yellow_threshold = cv.inRange(frame_HLS, (18, 73, 85), (57, 216, 255))
    red_threshold = cv.inRange(frame_HLS, (0, 101, 85), (15, 255, 255))
    white_threshold = cv.inRange(frame_gray, 160, 255)

    full_image = cv.add(yellow_threshold, cv.add(red_threshold, white_threshold))



    # blurred = cv.GaussianBlur(frame_threshold, (5,5),0)
    ## [while]

    ## [show]
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, white_threshold)
    cv.namedWindow('grayscale')
    cv.imshow('grayscale', full_image)
    ## [show]

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break