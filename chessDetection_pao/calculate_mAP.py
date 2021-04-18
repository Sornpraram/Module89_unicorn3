import cv2
import os
import shutil
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from termcolor import cprint
from models.dataset import Dataset
from models.utils import load_yolo_weights, detect_image, detect_realtime, detect_video
from models.configs import * 
from evaluate_mAP import get_mAP
import datetime

testset = Dataset('test')
model_checpoint = "./checkpoints/yolov4_tiny_all3_crop_aug_new_anchor2_512_loss_14.61_epoch_33"

if MODEL_TYPE == 'yolov4_tiny':
    from models.yolov4_tiny import Create_Model

elif MODEL_TYPE == 'zhumd1':
    from models.zhumd1 import Create_Model

model = Create_Model(input_size=INPUT_RESOLUTION, CLASSES=TRAIN_CLASSES)
model.load_weights(model_checpoint) 

get_mAP(model, testset, score_threshold=0.3, iou_threshold=0.45)