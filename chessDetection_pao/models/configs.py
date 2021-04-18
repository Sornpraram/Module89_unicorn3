# config for yolov4-tiny ONLY
MODEL_TYPE                  = "yolov4_tiny" # yolov4_tiny / zhumd1
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
# YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V4_TINY_WEIGHTS        = "./annotations/yolov4-tiny.weights"
YOLO_V3_TINY_WEIGHTS        = "./annotations/yolov3-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = True # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "./annotations/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
INPUT_RESOLUTION            = 512

# Train options
TRAIN_YOLO_TINY             = True
TRAIN_SAVE_BEST             = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES               = "./annotations/all3_crop_aug_names.txt" # "./model_data/license_plate_names.txt" share2aug
TRAIN_ANNOT_PATH            = "./annotations/all3_crop_aug_train.txt"
TRAIN_LOGDIR                = "log"
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{MODEL_TYPE}_all3_crop_aug_new_anchor2_{INPUT_RESOLUTION}"
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 8
TRAIN_INPUT_SIZE            = INPUT_RESOLUTION
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = False
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_FROM_CHECKPOINT_NAME  = "yolov4_best_loss_19.04_epoch_34"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 50

# TEST options
TEST_ANNOT_PATH             = "./annotations/all3_crop_aug_valid.txt" #"./model_data/license_plate_test.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = INPUT_RESOLUTION
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.5
TEST_IOU_THRESHOLD          = 0.45


#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    # YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
    #                         [[81,  82], [135, 169], [344, 319]],
    #                         [[0,    0], [0,     0], [0,     0]]] # YOLO tiny ใช้ scale แค่สองอัน? อันที่สามเลยเป็น 0

    # YOLO_ANCHORS            = [[[23,  29], [26, 34], [29, 37]],
    #                         [[33,     44], [37, 52], [44, 61]],
    #                         [[0,    0], [0,     0], [0,     0]]]               
    #

    # YOLO_ANCHORS            = [[[35,  37], [51, 51], [39, 70]],
    #                         [[81,  82], [135, 169], [344, 319]],
    #                         [[0,    0], [0,     0], [0,     0]]] # new_anchor

    YOLO_ANCHORS            = [[[16,  16], [28, 24], [20, 37]],
                            [[81,  82], [135, 169], [344, 319]],
                            [[0,    0], [0,     0], [0,     0]]]  # new_anchor2

    # YOLO_ANCHORS            = [[[35,  37], [51, 51], [39, 70]],
    #                         [[50,     54], [80, 90], [78, 120]],
    #                         [[0,    0], [0,     0], [0,     0]]]             


# if MODEL_TYPE                == "yolov4":
#     YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
#                             [[36,  75], [76,   55], [72,  146]],
#                             [[142,110], [192, 243], [459, 401]]]