import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from yolo_tiny_like_tf.dataset import Dataset
# from yolo_tiny_like_tf.yolov3_tiny_model import Create_Yolo, compute_loss
from yolo_tiny_like_tf.yolov4_tiny_model import Create_Yolo, compute_loss
from yolo_tiny_like_tf.utils import load_yolo_weights, detect_image
from yolo_tiny_like_tf.configs import *

yolo = Create_Yolo(input_size=INPUT_RESOLUTION, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov4_pao_loss_12.47_epoch_59") # use keras weights

image_path   = "./custom_dataset/valid/1.jpg"
video_path   = "./fibo_dataset/test.mp4"
image = detect_image(yolo, image_path, "", input_size=INPUT_RESOLUTION, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
# detect_video(yolo, video_path, "", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
# detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

# detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (800, 600))

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
