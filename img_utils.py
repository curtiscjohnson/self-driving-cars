import cv2
import numpy as np


def preprocess_image(BGRimg, sim=False, blackAndWhite=False):

    if blackAndWhite:
        originalImage = BGRimg
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

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
        blackImg[0:height//3,:,:] = (0, 0, 0)

        #black out bottom strip of image
        if not sim:
            blackImg[height - 1:height, :, :] = (0, 0, 0)

    return blackImg