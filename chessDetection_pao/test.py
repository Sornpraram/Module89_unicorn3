import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from termcolor import cprint
from models.dataset import Dataset
from models.utils import load_yolo_weights, detect_image, detect_realtime, detect_video
from models.configs import * 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        cprint("USING CPU INSTEAD", "red")
        print(e)

if MODEL_TYPE == 'yolov4_tiny':
    from models.yolov4_tiny import Create_Model

elif MODEL_TYPE == 'zhumd1':
    from models.zhumd1 import Create_Model

# ======================================================================================================================
# =============================================== Exiting Model ========================================================

# yolov4_tiny_all3_crop_aug_512_loss_13.71_epoch_44                     >>  yolov4-tiny / all3 / 512 / no transfer
# yolov4_tiny_all3_crop_aug_trans_512_loss_12.96_epoch_41               >>  yolov4-tiny / all3 / 512 / transfer
# yolov4_tiny_all3_crop_aug_trans_new_anchor_512_loss_36.93_epoch_40    >>  yolov4-tiny / all3 / 512 / transfer / new anchors
# yolov4_tiny_all3_crop_aug_new_anchor2_512_loss_14.61_epoch_33         >>  yolov4-tiny / all3 / 512 / no transfer / new anchors2
# zhumd1_all3_crop_aug_512_loss_14.43_epoch_39                          >>  zhumd1 / all3 / 512 / no transfer

# zhumd1_all2_crop_aug_512_loss_14.89_epoch_34                          >>  zhumd1 / all2 / 512 / no transfer



# ================= Load Model ================
model = Create_Model(input_size=INPUT_RESOLUTION, CLASSES=TRAIN_CLASSES)
model.load_weights("./checkpoints/yolov4_tiny_all3_crop_aug_new_anchor2_512_loss_14.61_epoch_33") 

# ============= Load Image/Videop =============
image_path   = "./dataset/TEST/img/test_all (17).jpg" #./dataset/aom_dataset/test/0430.jpg , wood_dataset/test/pic1.jpg C:\Module89_unicorn3\chessDetection\dataset\FOT_LABEl_ASSIST\batch1, C:\Module89_unicorn3\chessDetection\dataset\FOR_TEST_ONLY
video_path   = "./dataset/TEST/vid/test_all (2).mp4" # gold_silver_dataset/test/done22.mp4 , wood_dataset/test_vid1.mp4",

# ==== Detect Image ====
# image = detect_image(model, image_path, "", input_size=INPUT_RESOLUTION, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), crop=True)

# ==== Detect Video ====
detect_video(model, video_path, "", input_size=INPUT_RESOLUTION, show=True, rectangle_colors=(255,0,0), crop=True)

# ==== Detect Real-time ====
# detect_realtime(model, '', input_size=INPUT_RESOLUTION, show=True, rectangle_colors=(255, 0, 0), crop=False)
