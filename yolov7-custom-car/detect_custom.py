import argparse
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from copy import deepcopy

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# def detect(opt, device, model, img, save_img=False):

#     tStuff = time.time()
#     img = img[:,:,::-1].transpose(2,0,1)
#     img = np.ascontiguousarray(img)

#     t0 = time.time()
#     # for path, img, im0s, vid_cap in dataset:

#     img = torch.from_numpy(img).cuda()
#     img = img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     print(f"Time for stuff: {time.time() - tStuff}")
 
#     # Inference

#     t1 = time_synchronized()
#     tModel = time.time()
#     with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
#         pred = model(img, augment=opt.augment)[0]
#     print(f"Model time: {time.time() - tModel}")

#     # Apply NMS
#     tNMS = time.time()
#     pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#     # t3 = time_synchronized()
#     print(f"NMS time: {time.time() - tNMS}")

#     # Process detections
#     seen_signs = []
#     for i in range(pred[0].shape[0]):
#         print(i)
#         print(f"sign index : {int(pred[0][i,-1])}")



#     print(f'Done. ({time.time() - t0:.3f}s)')

#     return sign


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='1.jpg', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_false', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_false', help='don`t trace model')
#     opt = parser.parse_args()
#     # print(opt)
#     #check_requirements(exclude=('pycocotools', 'thop'))

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov7.pt']:
#                 detect()
#                 strip_optimizer(opt.weights)
#         else:
#             detect(opt)

def detect(opt, device, model, img, save_img=False):

    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # device = select_device(opt.device)

    # stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    t0 = time.time()
    # # # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    img = img[:,:,::-1].transpose(2,0,1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


    # Process detections
    seen_signs = []
    for i in range(pred[0].shape[0]):
        print(i)
        print(f"sign index : {int(pred[0][i,-1])}")
        seen_signs.append(int(pred[0][i,-1]))

    print(f'Done. ({time.time() - t0:.3f}s)')

    return seen_signs

