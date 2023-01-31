import cv2
import numpy as np
import random

def get_yellow_centers(bgr_img):
    """ gets centers of all yellows blobs on screen

    Args:
        img (_type_): BGR image from realsense camera

    Returns:
        float: list of tuples (cx, cy) center coordinates
    """

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

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
            if cy > hsv_img.shape[0]//3:
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
            cv2.circle(img, point, 7, (255, 0, 255), -1) 
            # args: img to draw on, point to draw, size of circle, color, line width (-1 defaults to fill)


def identify_possible_turns(shape, centers):
    LEFT_X_THRESH = shape[1] / 4
    RIGHT_X_THRESH = shape[1] *3/4
    Y_UPPER_THRESH = shape[0] *4.5/5
    Y_LOWER_THRESH = shape[0] *3/5
    turns = set()

    for center in centers:
        # check if there is a blob in the bottom center of the image
        if center[1] > Y_UPPER_THRESH and center[0] > LEFT_X_THRESH and center[0] < RIGHT_X_THRESH:
            return set()

        # check for left turn
        if center[0] < LEFT_X_THRESH and center[1] < Y_UPPER_THRESH and center[1] > Y_LOWER_THRESH:
            turns.add("left")

        # check for right turn
        if center[0] > RIGHT_X_THRESH and center[1] < Y_UPPER_THRESH and center[1] > Y_LOWER_THRESH:
            turns.add("right")
    
    return turns

def pick_turn(turns):
    if len(turns) == 1:
        turns.add("straight")

    return random.choice(list(turns))
