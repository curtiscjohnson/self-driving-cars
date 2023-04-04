import argparse
import sys
import time

import cv2
import torch

sys.path.append('..')
# sys.path.append('/fsg/hps22/self-driving-cars/')
import numpy as np
from detect_custom import detect
from models.experimental import attempt_load
from simple_pid import PID
from utils.torch_utils import select_device

from Arduino import Arduino
from img_utils import preprocess_image
from load_model_on_hardware_utils import setup_loading_model
from PID_Code.lightning_mcqueen import get_yellow_centers
from RealSense import *


# Class that operates as a state machine to keep track of signs and driving for the car
class StateMachine:
    def __init__(self):

        self.state = 'start car'
        # self.state = 'check for signs'
        self.lastSign = 'none'

        # Initialize YOLO Network
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov7-custom.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='2.jpg', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_false', help='don`t trace model')

        parser.add_argument('--control', type=str, default='rl', help='rl or pid')
        parser.add_argument('--yolo', type=bool, default=False, help='True: use yolo for sign recognition') 
        parser.add_argument('--speed', type=float, default=1.0, help='.8 to 3.0')
        parser.add_argument('--display', type=bool, default=True, help='True: show what camera is seeing')
        self.opt = parser.parse_args()

        self.device = select_device(self.opt.device)
        if self.opt.yolo:
            self.yolo_model = attempt_load(self.opt.weights, map_location=self.device).cuda()  # load FP32 model
            self.yolo_model(torch.zeros(1, 3, 480, 640).to(self.device).type_as(next(self.yolo_model.parameters())))  # run once

        # car params
        self.speed = self.opt.speed

        self.label_names = ['stop_sign','school_zone','construction_zone', 'do_not_pass','speed_limit','deer_crossing','rr_x','rr_circle','stop_light']

    def state_machine(self):

        while(True):

            if self.state == 'start car':
                self.start_car()

            elif self.state == 'drive':
                self.drive()

            elif self.state == 'check for signs' and self.opt.yolo:
                self.check_for_signs()
            
            elif self.state == 'terminate':
                break

    def start_car(self):
        if self.opt.control == 'rl':
            # load in RL model
            model_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016'
            zip_path = '/home/car/Desktop/self-driving-cars/sb3_models/local/curtis-20230325-124016/curtis-20230325-124016_model_800000_steps.zip'
            self.rl_model, self.config = setup_loading_model(model_path, zip_path)

        elif self.opt.control == 'pid':
            ## SETUP PID Controller
            self.pid = PID()
            self.pid.Ki = -.01*0
            self.pid.Kd = -.01*0
            self.pid.Kp = -30/300 #degrees per pixel
            frameUpdate = 1
            self.pid.sample_time = frameUpdate/30.0
            self.pid.output_limits = (-30,30)
            desXCoord = 640//2 #!hard coded to 640x480 
            self.pid.setpoint = desXCoord
        else:
            raise argparse.ArgumentTypeError("wrong contorl method")


        self.Car = Arduino("/dev/ttyUSB0", 115200)
        self.Car.zero(1440)
        self.Car.pid(1)          

        self.rs = RealSense("/dev/video2", RS_VGA)
        self.image = self.rs.getData(False)

        self.checkForSignsIndex = 0

        if self.opt.yolo:
            self.state = 'check for signs'
        else:
            self.state = 'drive'

    def drive(self):

        self.Car.drive(self.speed)

        # start1 = time.time()
        self.image = self.rs.getData(False)
        # getdata = time.time() - start1

        # display what camera sees
        if self.opt.display:
            cv2.namedWindow("car_raw", cv2.WINDOW_NORMAL)
            cv2.imshow("car_raw", self.image)

            if (cv2.waitKey(1) == ord('q')):
                cv2.destroyAllWindows()

        # print(self.image.shape)
        if self.opt.control == 'rl':
            # start = time.time()

            # prepare image to go into network
            preprocessedImg = preprocess_image(
                self.image,
                removeBottomStrip=True, #should always do this on hardware
                blackAndWhite=self.config["blackAndWhite"],
                addYellowNoise=False, #no need to add noise to real world
                use3imgBuffer=self.config["use3imgBuffer"],
                yellow_features_only=False,#self.config["yellow_features_only"],
                camera_resolution=self.config['camera_settings']['resolution']
            )
            # process_time = time.time() - start

            cv2.namedWindow("processed_img", cv2.WINDOW_NORMAL)
            cv2.imshow("processed_img", preprocessedImg)
            # print(preprocessedImg.shape)
            # networkImg = np.moveaxis(preprocessedImg, 2, 0)

            # camera_resolution = self.config['camera_settings']['resolution']
            # camera_resolution.reverse()
            # networkImg = cv2.resize(preprocessedImg, tuple(camera_resolution))
            # start = time.time()
            networkImg = np.moveaxis(preprocessedImg, 2, 0)
            # print(networkImg.shape)
            networkImg = np.moveaxis(networkImg, 1, 2)
            # print(networkImg.shape)
            # moveaxis = time.time() - start

            # get steering angle
            # start = time.time()
            with torch.no_grad():
                action_idx = self.rl_model(torch.from_numpy(networkImg/255).float().cuda()).max(0)[1].view(1,1)  
                angle = self.config["actions"][action_idx]
            # getaction = time.time() - start

        elif self.opt.control == 'pid':
            # prepare image to go into network
            centers = get_yellow_centers(self.image)
            if centers != "None":
                angle = self.pid(centers[-1][0])

        self.Car.steer(angle)
        # end = time.time()

        # print(f'getdata: {getdata}, process: {process_time}, moveaxis: {moveaxis}, getaction: {getaction}, loop time: {end-start1}')

        if self.checkForSignsIndex % 7 == 0 and self.opt.yolo:
            self.state = 'check for signs'

    def check_for_signs(self):

        # imgToNetwork = self.preprocess_image(self.image)
        # cv2.imwrite('/home/car/Desktop/self-driving-cars/yolov7_sign_detection_copy/1.jpg', self.image)
        sign = self.get_current_sign()

        if sign == 'stop_sign':
            self.Car.drive(0)
            if self.lastSign != 'stop_sign':
                self.lastSign = 'stop_sign'
                time.sleep(2)
        elif sign == 'school_zone':
            self.Car.drive(0)
            if self.lastSign != 'school_zone':
                self.lastSign = 'school_zone'
                self.Car.music(4)
        elif sign == 'construction_zone':
            self.Car.drive(0)
            if self.lastSign != 'construction_zone':
                self.lastSign = 'construction_zone'
                self.Car.music(2)
        elif sign == 'do_not_pass':
            self.Car.drive(0)
            if self.lastSign != 'do_not_pass':
                self.lastSign = 'do_not_pass'
                self.Car.music(0)
        if sign == 'speed_limit':
            self.Car.drive(0)
            if self.lastSign != 'speed_limit':
                self.lastSign = 'speed_limit'
                self.Car.music(1)
        elif sign == 'deer_crossing':
            self.Car.drive(0)
            if self.lastSign != 'deer_crossing':
                self.lastSign = 'deer_crossing'
                self.Car.music(5)
        elif sign == 'rr_x':
            self.Car.drive(0)
            if self.lastSign != 'rr_x':
                self.lastSign = 'rr_x'
                self.Car.music(3)
        elif sign == 'rr_circle':
            self.Car.drive(0)
            if self.lastSign != 'rr_circle':
                self.lastSign = 'rr_circle'
                self.Car.music(3)
        elif sign == 'stop_light':
            self.Car.drive(0)
            if self.lastSign != 'stop_light':
                self.lastSign = 'stop_light'
                self.Car.music(7)

        self.state = 'drive'

    def get_current_sign(self):

        loop_time = time.time()
        # img = cv2.imread('8.jpg')
        img = self.image

        with torch.no_grad():        
            signs_seen, signs_seen_location, signs_seen_confidence = detect(self.opt, self.device, self.yolo_model, img)

        labels_seen, labels_loc, labels_confidence = self.cleanup_network_output(signs_seen, signs_seen_location, signs_seen_confidence)
        print(f"\nSign: {labels_seen}, Location: {labels_loc}, Confidence: {labels_confidence} in {time.time() - loop_time} seconds.")

        # if labels_seen != 'none':
        #     print(f"Sign Bounding Box Area: {self.calculate_bounding_box_area(labels_loc)}")
        #     self.draw_bounding_box(img, labels_loc)

        return labels_seen

    def cleanup_network_output(self, signs_seen, signs_seen_location, signs_seen_confidence):

        # ['stop_sign','school_zone','construction_zone', 'do_not_pass','speed_limit','deer_crossing','rr_x','rr_circle','stop_light']
        minimumArea = [9500., 18000., 13500., 4500., 6500., 11500., 6000., 12500., 10000.]

        # signs_seen is a list of indexes of signs that were seen in the image
        if len(signs_seen) > 0:

            bigEnoughSigns = []
            bigEnoughSignsLocations = []
            bigEnoughSignsConfidence = []

            for i in range(0, len(signs_seen)):

                sign_loc = signs_seen_location[i].cpu().numpy()
                sign_area = self.calculate_bounding_box_area(sign_loc)

                if sign_area > minimumArea[signs_seen[i]]:
                    bigEnoughSigns.append(self.label_names[signs_seen[i]])
                    bigEnoughSignsLocations.append(signs_seen_location[i].cpu().numpy())
                    bigEnoughSignsConfidence.append(signs_seen_confidence[i].cpu().numpy())

            if not bigEnoughSigns:
                print('here')
                return 'none', 'none', 'none'

            signConfidenceIndex = np.argmax(bigEnoughSignsConfidence)

            return bigEnoughSigns[signConfidenceIndex], bigEnoughSignsLocations[signConfidenceIndex], bigEnoughSignsConfidence[signConfidenceIndex]

        else:
            return 'none', 'none', 'none'

    def calculate_bounding_box_area(self, labels_loc):

        length = np.abs(labels_loc[0] - labels_loc[2])
        width = np.abs(labels_loc[1] - labels_loc[3])

        return length * width

    def draw_bounding_box(self, img, labels_loc):

        start_point1 = (int(labels_loc[0]), int(labels_loc[1]))
        end_point1 = (int(labels_loc[2]), int(labels_loc[1]))
        start_point2 = (int(labels_loc[2]), int(labels_loc[1]))
        end_point2 = (int(labels_loc[2]), int(labels_loc[3]))
        start_point3 = (int(labels_loc[2]), int(labels_loc[3]))
        end_point3 = (int(labels_loc[0]), int(labels_loc[3]))
        start_point4 = (int(labels_loc[0]), int(labels_loc[3]))
        end_point4 = (int(labels_loc[0]), int(labels_loc[1]))

        imageWithBoundingBox = cv2.line(img, start_point1, end_point1, (255, 0, 0), 2) 
        imageWithBoundingBox = cv2.line(img, start_point2, end_point2, (255, 0, 0), 2) 
        imageWithBoundingBox = cv2.line(img, start_point3, end_point3, (255, 0, 0), 2) 
        imageWithBoundingBox = cv2.line(img, start_point4, end_point4, (255, 0, 0), 2)

        cv2.imshow("car", imageWithBoundingBox)
        if (cv2.waitKey(1) == ord('q')):
            cv2.destroyAllWindows()

if __name__ == "__main__":

    stateMachine = StateMachine()
    stateMachine.state_machine()