# -*- coding: utf-8 -*-
"""road_lane_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w6xv6v5xuQPmMJMMQ5MZnGlC6F9bsi7Q
"""

#ROAD LANE DETECTION

import cv2
import numpy as np

def grey(image):
  #convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

#outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image,50,150)
    return edges

def region(image):
    # only focus bottom half of the screen
    height, width = image.shape
    #isolate the gradients that correspond to the lane lines
    polygon = np.array([[
        (0, height * 2/3),
        (width, height * 2/3),
        (width, height),
        (0, height),
    ]], np.int32)

    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, polygon, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    #make sure dict isn't empty
    if lines is not None:
        for line in lines.values():
            x1, y1, x2, y2 = line
            #draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

def average(image, lines):
    left = []
    right = []
    # print(f"Averaging {len(lines)} lines")
    if lines is not None:
      for line in lines:
        # print(f"\nLine (x0,y0,x1,y1): {line}")
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(f"slope:{parameters[0]}, y_int:{parameters[1]}")
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
            # print("Classified as left line")
        else:
            right.append((slope, y_int))
            # print("Classified as right line")

            
    #takes average among all the columns (column0: slope, column1: y_int)
    detected_lines = {}
    if len(left) > 0:
        left_avg = np.average(left, axis=0)
        # print(f"Left Average Slope/Yint:{left_avg}")
        left_line = make_points(image, left_avg)
        detected_lines['left'] = left_line

    if len(right) > 0:
        right_avg = np.average(right, axis=0)
        # print(f"Right Average Slope/Yint:{right_avg}")
        right_line = make_points(image, right_avg)
        detected_lines['right'] = right_line

    #create lines based on averages calculates
    return detected_lines

def make_points(image, average):
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (2/3))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def get_white_lane_lines(image):
    # filter for blue lane lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # show_image("img", hsv)
    lower_white = np.array([0,0,155])
    upper_white = np.array([113,255,205])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    return mask

def detect_lanes(img):
    # cv2.imshow('original frame', img)
    copy = np.copy(img)
    #threshold on white first
    white = get_white_lane_lines(copy)
    # cv2.imshow('white lane lines', white)
    edges = cv2.Canny(white,50,150)
    isolated = region(edges)
    # cv2.imshow("edges", edges)
    # cv2.imshow("bottom third", isolated)


    #DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array, 
    lines = cv2.HoughLinesP(isolated, 1, np.pi/180, 10, np.array([]), minLineLength=40, maxLineGap=5)
    # print(f"Lines found:\n {lines}")
    averaged_lines = average(copy, lines)
    # print(f"Averaged Lines:\n {averaged_lines}")
    # black_lines = display_lines(copy, averaged_lines)
    # #taking wighted sum of original image and lane lines image
    # lanes = cv2.addWeighted(copy, 0.2, black_lines, 1, 0)
    # cv2.imshow("lanes", lanes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return averaged_lines #dict of left or right lines as keys and values of (x0,y0,x1,y1) defining line

# for filename in glob.glob('./sim_testing_images/*.png'):
#     img = cv2.imread(filename)
#     plt.imshow(img)
#     lanes = detect_lanes(img)
#     print(f"LANES: {lanes}")

def get_road(img):
    lanes = detect_lanes(img)

    width = img.shape[1]
    height = img.shape[0]

    polygon = np.array([[
        (0, height * 2/3),
        (width, height * 2/3),
        (width, height),
        (0, height),
    ]], np.int32)
    # print(f"shape: {polygon.shape}")
    # pt1 ------------- pt2
    #  |                 |
    #  |                 |
    # pt4 ------------- pt3

    for side, points in lanes.items():
        if side == "left":
            polygon[0, 0, :] = points[-2:]
            polygon[0, 3, :] = points[:2]
        elif side == "right":
            polygon[0, 1, :] = points[-2:]
            polygon[0, 2, :] = points[:2]

    #create a mask (triangle that isolates the region of interest in our image)
    mask = np.zeros(img.shape, dtype="uint8")*255
    mask = cv2.fillPoly(mask, polygon, color=(255, 255, 255))

    # print(f"mask shape: {mask.shape}")
    # print(f"img shape: {img.shape}")
    masked_img = cv2.bitwise_and(img, mask)

    # masked = cv2.addWeighted(masked, 1, img, 1, 0)
    # cv2.imshow("FINAL VIEW", masked_img)

    return masked_img