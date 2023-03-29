import sys
import argparse
import torch
import cv2
sys.path.append('../..')
sys.path.append('/fsg/hps22/self-driving-cars/')
# sys.path.append('/fsg/hps22/groups/self-driving/YOLO_Data/haleyTry/yolov7-custom/')
# from detect_lane import grey
from detect_custom import detect
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from models.experimental import attempt_load
from Arduino import Arduino

# Class that operates as a state machine to keep track of signs and driving for the car
class StateMachine:
    def __init__(self):

        self.state = 'start car'

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov7-custom.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='1.jpg', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
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

        # Initialize YOLO Network
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.opt.weights, map_location=device)  # load FP32 model

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

        # start camera, car, everything needed to drive here
        self.Car = Arduino("/dev/ttyUSB0", 115200)

        self.state = 'drive'

    def drive(self):

        # set car speed

        # get wheel angle

        # start car

        self.state = 'check for signs'

    def check_for_signs(self):

        imgFromCar = cv2.imread('/fsg/hps22/groups/self-driving/YOLO_Data/haleyTry/yolov7-custom/2.jpg')
        imgToNetwork = cv2.imwrite('/fsg/hps22/groups/self-driving/YOLO_Data/haleyTry/yolov7-custom/1.jpg', imgFromCar)
        sign = self.get_current_sign()

        print('\n\n\n\n\n\n\n\n',sign,'\n\n\n\n\n\n\n\n\n')

        # if sign == 'stop_sign':
        #     # sound
        # elif sign == 'school_zone':
        #     # sound
        # elif sign == 'construction_zone':
        #     # sound
        # elif sign == 'do_not_pass':
        #     # sound
        if sign == 'speed_limit':
            self.Car.music(self, 5)
        # elif sign == 'deer_crossing':
        #     # sound
        # elif sign == 'rr_x':
        #     # sound
        # elif sign == 'rr_circle':
        #     # sound
        # elif sign == 'stop_light':
        #     # sound

        self.state = 'drive'

    def get_current_sign(self):

        with torch.no_grad():
            if self.opt.update:  # update all models (to fix SourceChangeWarning)
                for self.opt.weights in ['yolov7.pt']:
                    sign = detect(self.opt)
                    strip_optimizer(self.opt.weights)
            else:
                sign = detect(self.opt)

        sign = self.cleanup_network_output(sign)

        return sign 

    def cleanup_network_output(self, sign):

        if "speed_limit" in sign:
            cleanSign = 'speed_limit'
        else:
            cleanSign = 'none'

        return cleanSign


if __name__ == "__main__":

    stateMachine = StateMachine()
    stateMachine.state_machine()