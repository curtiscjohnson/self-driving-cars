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

def detect(opt, device, model, img, save_img=False):

    t0 = time.time()

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
    # print(pred)

    # Process detections
    seen_signs_index = []
    seen_signs_location = []
    seen_signs_confidence = []
    for i in range(pred[0].shape[0]):
        # print(i)
        # print(f"sign index : {int(pred[0][i,-1])}")
        seen_signs_index.append(int(pred[0][i,-1]))
        seen_signs_location.append(pred[0][i, 0:4])
        seen_signs_confidence.append(pred[0][i, 4])

    # print(f'Done. ({time.time() - t0:.3f}s)')

    return seen_signs_index, seen_signs_location, seen_signs_confidence

