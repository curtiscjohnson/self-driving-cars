import cv2
import numpy as np


def preprocess_image(BGRimg, removeBottomStrip=False, blackAndWhite=False, addYellowNoise=False, use3imgBuffer=False):

    BGRimg = cv2.bilateralFilter(BGRimg,5,75,75) #theoretically good at removing noise but keeping sharp lines.


    #? may want to try adaptive thresholding on hardware. Might help with whites? 
    #? https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=image-,Adaptive%20Thresholding,-In%20the%20previous

    #! BGRimg comes in as height x width x channels
    if addYellowNoise:
        # Add random yellow squares to the image
        ypix, xpix, channels = BGRimg.shape
        num_randpoints = 2
        x_points = np.random.randint(xpix//3, xpix, size=num_randpoints)
        y_points = np.random.randint(ypix//3, ypix, size=num_randpoints)
        for i in range(0,num_randpoints):
            size = np.random.randint(0,2)
            point = (x_points[i], y_points[i])
            point_2 = (x_points[i]+size, y_points[i]+size)

            #not sure why this is needed, but works.
            #see https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
            BGRimg = np.ascontiguousarray(BGRimg, dtype=np.uint8)

            cv2.rectangle(BGRimg, point, point_2, (0, 255, 255), -1)

    if blackAndWhite:
        originalImage = BGRimg
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("in_preprocess", cv2.WINDOW_NORMAL)
        # cv2.imshow("in_preprocess", grayImage)
        
        # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY) #for sim
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 120, 255, cv2.THRESH_BINARY) #for hw


        if use3imgBuffer:
            return blackAndWhiteImage

        blackImg = np.zeros(BGRimg.shape, dtype = np.uint8)
        blackImg[:,:,0] = blackAndWhiteImage
        blackImg[:,:,1] = blackAndWhiteImage
        blackImg[:,:,2] = blackAndWhiteImage


        # cv2.imshow("car", blackImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        HSVimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HSV)
        HSVimg = cv2.bilateralFilter(HSVimg,9,75,75)
        # HSVimg = cv2.GaussianBlur(HSVimg, (5,5),0)
        blackImg = np.zeros(HSVimg.shape, dtype = np.uint8)
    
        # make true yellow
        lower_yellow = np.array([15,89,124])
        upper_yellow = np.array([100,255,255])
        mask=cv2.inRange(HSVimg,lower_yellow,upper_yellow)
        blackImg[mask>0] = (0,255,255)

        # make true red
        lower_red = np.array([0,50,146])
        upper_red = np.array([5,255,255])
        mask=cv2.inRange(HSVimg,lower_red,upper_red)
        blackImg[mask>0] = (0,0,255)

    # black out top 1/3 of image
    height, width, depth = blackImg.shape
    blackImg[0:height//2,:,:] = (0, 0, 0)

    #black out bottom strip of image
    if not removeBottomStrip:
        blackImg[height - 1:height, :, :] = (0, 0, 0)

    return blackImg