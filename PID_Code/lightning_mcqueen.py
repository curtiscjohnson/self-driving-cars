import cv2
import numpy as np
import random

def get_yellow_centers(BGRimg):
    """ gets centers of all yellows blobs on screen

    Args:
        img (_type_): BGR image from realsense camera

    Returns:
        float: list of tuples (cx, cy) center coordinates
    """
    frame_HLS = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HLS)
    # hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Get grayscale image with only centerline (yellow colors)
    lower_yellow = np.array([14,95,147])
    upper_yellow = np.array([100,255,255])
    centerline_gray_img = cv2.inRange(frame_HLS, (18, 95, 91), (57, 216, 255)) # get only yellow colors in image



    # centerline_gray_img = cv2.inRange(frame_HLS, (18, 130, 91), (57, 216, 255)) # get only yellow colors in image
    blurred = cv2.GaussianBlur(centerline_gray_img, (5,5),0)

    # Get Contours for center line blobs
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if cy > frame_HLS.shape[0]*3.25/5 and cy < frame_HLS.shape[0]*6/7: #where to look for centers
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
    
    return turns, [LEFT_X_THRESH, RIGHT_X_THRESH, Y_UPPER_THRESH, Y_LOWER_THRESH]

def pick_turn(turns):
    if len(turns) == 1:
        turns.add("straight")

    return random.choice(list(turns))

def get_buffer_avg(x_values):

    return np.average(x_values, weights=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.075, 0.075, 0.05, 0.05])

def wait_for_red_lights(realsense, sign):
    red_lights_on = True
    while red_lights_on:
        BGRimg = realsense.getData(False)
        height, width, depth = BGRimg.shape
        BGRimg[height // 2:, :, :] = (0, 0, 0)
        BGRimg[:,:width//3:, :] = (0,0,0)
        # black out bottom strip of image
        BGRimg[height - 35 : height, :, :] = (0, 0, 0)

        BGRimg = cv2.bilateralFilter(
        BGRimg, 3, 50, 50
        )  # theoretically good at removing noise but keeping sharp lines.

        frame_HLS = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HLS)
        if sign == 'rr_x':
            red_threshold = cv2.inRange(frame_HLS, (14, 0, 181), (33, 255, 255))
            cv2.namedWindow("red_lights", cv2.WINDOW_NORMAL)
            cv2.imshow("red_lights", red_threshold)

            if (cv2.waitKey(1) == ord('q')):
                cv2.destroyAllWindows()
            if np.sum(red_threshold//255) > 20:
                red_lights_on = True
            else:
                red_lights_on = False
        elif sign == 'rr_circle':
            red_threshold = cv2.inRange(frame_HLS, (14, 0, 205), (33, 255, 255))
            cv2.namedWindow("red_lights", cv2.WINDOW_NORMAL)
            cv2.imshow("red_lights", red_threshold)

            if (cv2.waitKey(1) == ord('q')):
                cv2.destroyAllWindows()
            if np.sum(red_threshold//255) > 50:
                red_lights_on = True
            else:
                red_lights_on = False


    