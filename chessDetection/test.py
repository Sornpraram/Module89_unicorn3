import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from yolo_tiny_like_tf.dataset import Dataset
from yolo_tiny_like_tf.yolov3_tiny_model import Create_Yolo, compute_loss
# from yolo_tiny_like_tf.yolov4_tiny_model import Create_Yolo, compute_loss
from yolo_tiny_like_tf.utils import load_yolo_weights, detect_image
from yolo_tiny_like_tf.configs import *

yolo = Create_Yolo(input_size=INPUT_RESOLUTION, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_pao_loss_53.99_epoch_2") # use keras weights

image_path   = "./custom_dataset/valid/3aafc2d38807dddd1b43a54cb70f500d_jpg.rf.7a1acfea51aff18b554e96c49beafb78.jpg"
image = detect_image(yolo, image_path, "", input_size=INPUT_RESOLUTION, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (800, 600))

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()