import sys
import argparse
import torch
import cv2
import time
sys.path.append('..')
sys.path.append('/fsg/hps22/self-driving-cars/')
from detect_custom import detect
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from models.experimental import attempt_load
from Arduino import Arduino
from RealSense import *
import PID_Code.lightning_mcqueen as lm
from simple_pid import PID
import numpy as np

# Class that operates as a state machine to keep track of signs and driving for the car
class StateMachine:
    def __init__(self):

        # self.state = 'start car'
        self.state = 'check for signs'
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
        self.opt = parser.parse_args()

        self.device = select_device(self.opt.device)
        self.model = attempt_load(self.opt.weights, map_location=self.device).cuda()  # load FP32 model
        self.model(torch.zeros(1, 3, 480, 640).to(self.device).type_as(next(self.model.parameters())))  # run once

        # car params
        self.speed = 0.8

        self.label_names = ['stop_sign','school_zone','construction_zone', 'do_not_pass','speed_limit','deer_crossing','rr_x','rr_circle','stop_light']

    def state_machine(self):

        while(True):

            if self.state == 'start car':
                self.start_car()

            elif self.state == 'drive':
                self.drive()

            elif self.state == 'check for signs':
                self.check_for_signs()
            
            elif self.state == 'terminate':
                break

    def start_car(self):

        self.Car = Arduino("/dev/ttyUSB0", 115200)
        self.Car.zero(1440)
        self.Car.pid(1)          

        enableDepth = True
        self.rs = RealSense("/dev/video2", RS_VGA, enableDepth)
        self.image = self.rs.getData(False)

        self.pid = PID()
        self.pid.Ki = -.01*0
        self.pid.Kd = -.01*0
        self.pid.Kp = -30/250
        self.frameUpdate = 1
        self.pid.sample_time = self.frameUpdate/30.0
        self.pid.output_limits = (-30,30)
        desXCoord = self.image.shape[0]*3/5
        self.pid.setpoint = desXCoord
        self.i = 0

        self.checkForSignsIndex = 0

        self.state = 'check for signs'

    def drive(self):

        self.Car.drive(self.speed)

        self.image = self.rs.getData(False)
        if self.i % self.frameUpdate == 0:

            self.i = 0

            centers = lm.get_yellow_centers(self.image)

            if centers != "None":
                blobToFollowCoords = centers[-1]
                blobX = blobToFollowCoords[0]
                angle = self.pid(blobX)
                self.Car.steer(angle)
        self.i+=1
        self.checkForSignsIndex += 1

        if self.checkForSignsIndex % 7 == 0:
            self.state = 'check for signs'

    def preprocess_image(self, image):

        # newImage = image[:, 320:]
        newImage = cv2.resize(image, (128, 128))

        return newImage

    def check_for_signs(self):

        # imgToNetwork = self.preprocess_image(self.image)
        # cv2.imwrite('/home/car/Desktop/self-driving-cars/yolov7_sign_detection_copy/1.jpg', self.image)
        sign = self.get_current_sign()

        # if sign == 'stop_sign':
        #     self.Car.drive(0)
        #     if self.lastSign != 'stop_sign':
        #         self.lastSign = 'stop_sign'
        #         time.sleep(2)
        # elif sign == 'school_zone':
        #     self.Car.drive(0)
        #     if self.lastSign != 'school_zone':
        #         self.lastSign = 'school_zone'
        #         self.Car.music(4)
        # elif sign == 'construction_zone':
        #     self.Car.drive(0)
        #     if self.lastSign != 'construction_zone':
        #         self.lastSign = 'construction_zone'
        #         self.Car.music(2)
        # elif sign == 'do_not_pass':
        #     self.Car.drive(0)
        #     if self.lastSign != 'do_not_pass':
        #         self.lastSign = 'do_not_pass'
        #         self.Car.music(0)
        # if sign == 'speed_limit':
        #     self.Car.drive(0)
        #     if self.lastSign != 'speed_limit':
        #         self.lastSign = 'speed_limit'
        #         self.Car.music(1)
        # elif sign == 'deer_crossing':
        #     self.Car.drive(0)
        #     if self.lastSign != 'deer_crossing':
        #         self.lastSign = 'deer_crossing'
        #         self.Car.music(5)
        # elif sign == 'rr_x':
        #     self.Car.drive(0)
        #     if self.lastSign != 'rr_x':
        #         self.lastSign = 'rr_x'
        #         self.Car.music(3)
        # elif sign == 'rr_circle':
        #     self.Car.drive(0)
        #     if self.lastSign != 'rr_circle':
        #         self.lastSign = 'rr_circle'
        #         self.Car.music(3)
        # elif sign == 'stop_light':
        #     self.Car.drive(0)
        #     if self.lastSign != 'stop_light':
        #         self.lastSign = 'stop_light'
        #         self.Car.music(7)

        # self.state = 'drive'

    def get_current_sign(self):

        loop_time = time.time()
        img = cv2.imread('4.jpg')
        # img = self.image

        with torch.no_grad():        
            signs_seen, signs_seen_location, signs_seen_confidence = detect(self.opt, self.device, self.model, img)

        labels_seen, labels_loc, labels_confidence = self.cleanup_network_output(signs_seen, signs_seen_location, signs_seen_confidence)
        print(f"\nSign: {labels_seen}, Location: {labels_loc}, Confidence: {labels_confidence} in {time.time() - loop_time} seconds.")

        print(f"Sign Bounding Box Area: {self.calculate_bounding_box_area(labels_loc)}")
        self.draw_bounding_box(img, labels_loc)

        return labels_seen

    def cleanup_network_output(self, signs_seen, signs_seen_location, signs_seen_confidence):

        # ['stop_sign','school_zone','construction_zone', 'do_not_pass','speed_limit','deer_crossing','rr_x','rr_circle','stop_light']
        minimumArea = [5000., 1000., 1000., 1000., 1000., 1000., 200., 1000., 1000.]

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