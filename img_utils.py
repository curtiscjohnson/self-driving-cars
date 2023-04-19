import cv2
import numpy as np


def preprocess_image(
    BGRimg,
    removeBottomStrip=False,
    blackAndWhite=False,
    addYellowNoise=False,
    use3imgBuffer=False,
    yellow_features_only=False,
    column_mask=False,
    camera_resolution=(100,100)
):
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", BGRimg)
    # cv2.waitKey(0)

    # crop top of image
    height, width, depth = BGRimg.shape
    BGRimg = BGRimg[height//2:height - 35, :, :]

    BGRimg = cv2.resize(BGRimg, (147,240))

    # assert BGRimg.shape == (640,480), f'Image size is _{BGRimg.shape}, not (640, 480)'
    BGRimg = cv2.bilateralFilter(
        BGRimg, 3, 50, 50
    )  # theoretically good at removing noise but keeping sharp lines.


    # cv2.namedWindow("filtered", cv2.WINDOW_NORMAL)
    # cv2.imshow("filtered", BGRimg)
    # cv2.waitKey(0)

    # ? may want to try adaptive thresholding on hardware. Might help with whites?
    # ? https://docs.opencv2.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=image-,Adaptive%20Thresholding,-In%20the%20previous


    if column_mask:
        frame_HLS = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HLS)
        frame_HSV = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HSV)
        blackImg = np.zeros(BGRimg.shape, dtype=np.uint8)
        # print(blackImg.shape)
        # print(blackAndWhiteImage.shape)
        yellow_threshold = cv2.inRange(frame_HLS, (18, 73, 85), (57, 216, 255))
        red_threshold = cv2.inRange(frame_HLS, (0, 101, 85), (15, 255, 255))

        blurred = cv2.GaussianBlur(frame_HSV, (5,5),0)
        white_lines = cv2.inRange(blurred, (0, 0, 146), (180, 37, 255))
        last_y_vals = np.where(np.count_nonzero(white_lines, axis=0)==0, -1, (white_lines.shape[0]-1) - np.argmax(white_lines[::-1,:]!=0, axis=0))
        row_indeces, col_indeces = np.indices(white_lines.shape)
        mask = row_indeces <= last_y_vals.reshape(1,-1)
        white_lines[mask] = 255

        blackImg[:, :, 0] = yellow_threshold
        blackImg[:, :, 1] = red_threshold
        blackImg[:, :, 2] = white_lines

        return cv2.resize(blackImg, tuple(camera_resolution))

    #! BGRimg comes in as height x width x channels
    if addYellowNoise:
        # Add random yellow squares to the image
        ypix, xpix, channels = BGRimg.shape
        num_randpoints = 200
        x_points = np.random.randint(xpix // 3, xpix, size=num_randpoints)
        y_points = np.random.randint(ypix // 3, ypix, size=num_randpoints)
        for i in range(0, num_randpoints):
            size = np.random.randint(0, 1)
            point = (x_points[i], y_points[i])
            point_2 = (x_points[i] + size, y_points[i] + size)

            # not sure why this is needed, but works.
            # see https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
            BGRimg = np.ascontiguousarray(BGRimg, dtype=np.uint8)

            cv2.rectangle(BGRimg, point, point_2, (0, 255, 255), -1)

    if blackAndWhite:
        # if yellow_features_only:
        #     # take only yellow to grayscale operation - maybe do adaptive thresholding here?
        #     HLSimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HLS)
        #     lower_yellow = np.array([18, 73, 85])
        #     upper_yellow = np.array([57, 216, 255])
        #     blackAndWhiteImage = cv2.inRange(HLSimg, lower_yellow, upper_yellow)
        #     # cv2.namedWindow("yellow_thresholded", cv2.WINDOW_NORMAL)
        #     # cv2.imshow("yellow_thresholded", blackAndWhiteImage)
        #     # cv2.waitKey(0)
        # else:
        #     # take all color to grayscale
        #     originalImage = BGRimg
        #     grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        #     # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY) #for sim
        #     (thresh, blackAndWhiteImage) = cv2.threshold(
        #         grayImage, 120, 255, cv2.THRESH_BINARY
        #     )  # for hw

        # cv2.namedWindow("black and white", cv2.WINDOW_NORMAL)
        # cv2.imshow("black and white", blackAndWhiteImage)
        # cv2.waitKey(0)


        frame_HLS = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HLS)
        frame_gray = cv2. cvtColor(BGRimg, cv2.COLOR_BGR2GRAY)

        yellow_threshold = cv2.inRange(frame_HLS, (18, 130, 91), (57, 216, 255))
        red_threshold = cv2.inRange(frame_HLS, (0, 101, 85), (15, 255, 255))
        white_threshold = cv2.inRange(frame_gray, 160, 255)

        blackAndWhiteImage = cv2.add(yellow_threshold, cv2.add(red_threshold, white_threshold))

        if use3imgBuffer:
            return blackAndWhiteImage

        blackImg = np.zeros(BGRimg.shape, dtype=np.uint8)
        # print(blackImg.shape)
        # print(blackAndWhiteImage.shape)
        blackImg[:, :, 0] = blackAndWhiteImage
        blackImg[:, :, 1] = blackAndWhiteImage
        blackImg[:, :, 2] = blackAndWhiteImage

        # blackImg[:, :] = blackAndWhiteImage
        # blackImg[1, :, :] = blackAndWhiteImage
        # blackImg[2, :, :] = blackAndWhiteImage

        # cv2.imshow("car", blackImg)
        # cv2.destroyAllWindows()
        # print("did what we want")
    else:
        HSVimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2HSV)
        HSVimg = cv2.bilateralFilter(HSVimg, 9, 75, 75)
        # HSVimg = cv2.GaussianBlur(HSVimg, (5,5),0)
        blackImg = np.zeros(HSVimg.shape, dtype=np.uint8)

        # make true yellow
        lower_yellow = np.array([15, 89, 124])
        upper_yellow = np.array([100, 255, 255])
        mask = cv2.inRange(HSVimg, lower_yellow, upper_yellow)
        blackImg[mask > 0] = (0, 255, 255)

        # make true red
        lower_red = np.array([0, 50, 146])
        upper_red = np.array([5, 255, 255])
        mask = cv2.inRange(HSVimg, lower_red, upper_red)
        blackImg[mask > 0] = (0, 0, 255)

    # return cv2.resize(blackImg, 
    # print(blackImg.shape)
    # print((3, *camera_resolution))
    # return cv2.resize(blackImg, (3, *camera_resolution))
    # print(camera_resolution)
    # print(*camera_resolution)
    # print((*camera_resolution))
    # camera_resolution.reverse()
    # print(camera_resolution)
    # print(blackImg.shape)
    # return blackImg
    return cv2.resize(blackImg, tuple(camera_resolution))

