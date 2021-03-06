import cv2
import os
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from termcolor import cprint
from models.dataset import Dataset
from models.utils import load_yolo_weights, detect_image, save_loss
from evaluate_mAP import get_mAP
from models.configs import *

if MODEL_TYPE == 'yolov4_tiny':
    from models.yolov4_tiny import Create_Model, compute_loss

elif MODEL_TYPE == 'zhumd1':
    from models.zhumd1 import Create_Model, compute_loss

def main(): 
    global TRAIN_FROM_CHECKPOINT
    #====================================== Tensorflow-GPU Detector =======================================

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    if len(gpus) > 0:
        
        cprint("using GPU", "green")
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    else:
        print("using CPU")
    #======================================== Tensorboard summary ===========================================
    if os.path.exists(TRAIN_LOGDIR): 
        shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    #====================================== Calculate Training Step ========================================

    trainset = Dataset('train')
    testset = Dataset('test')

    steps_per_epoch = len(trainset) # iterations or num_batch
    print(steps_per_epoch)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) # create tf variable to hold value of global step
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch # 2 * 26 = 52 Stepแรก จะค่อยๆทำการเพิ่ม lr
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    #============================ Create Model & Load Checpoint & Init Optimizer =========================================

    if TRAIN_TRANSFER:
        if MODEL_TYPE == "yolov4_tiny":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS

        print(Darknet_weights)
        Darknet = Create_Model(input_size=INPUT_RESOLUTION, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    model = Create_Model(input_size=INPUT_RESOLUTION, training=True, CLASSES=TRAIN_CLASSES)

    if TRAIN_FROM_CHECKPOINT:
        try:
            model.load_weights(f"./checkpoints/{TRAIN_FROM_CHECKPOINT_NAME}")
            cprint(f"Train from checkpoint >>> {TRAIN_FROM_CHECKPOINT_NAME} ", "green")
        except:
            cprint("Load Checkpoint Error >>> Train from zero instead", "red")

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    model.layers[i].set_weights(layer_weights)
                    cprint(f"Transfer Learning >>> {YOLO_V4_TINY_WEIGHTS} ", "green")
                except:
                    print("skipping", model.layers[i].name)

    if not TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        cprint(f"Train from Zero >>> {MODEL_TYPE} ", "blue")

    optimizer = tf.keras.optimizers.Adam()

    # =============================================================== train step ===================================================================
    
    def train_step(image_data, target):

        with tf.GradientTape() as tape: # ใช้คำนวณ GradiantDesent ของ trainable variable ทุกตัว

            pred_result = model(image_data, training=True) # Feed image เข้า Model. # Return
            giou_loss = 0  # loss function ()
            conf_loss = 0  # loss function ()
            prob_loss = 0  # loss function ()

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss # ค่า loss total เป็นผลรวมของ loss

            gradients = tape.gradient(total_loss, model.trainable_variables) # คำนวณ Gradiant tape.gradient(target, sources) return eagerTensor
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # ========================================= update learning rate (warmup algorithm) ============================================

            global_steps.assign_add(1)
            if global_steps < warmup_steps:   # and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT   # (global_step) / 52 *  0.0001  # ใน 52 step แรก lr จะค่อยๆเพิ่มขึ้น จากนั้นจะคงที่
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(  # 0.000001 + [0.5 * (0.0001 - 0.000001) * some funcking algorithm]
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))) # 0.000001 เซ็ตค่า lr ต่ำสุดที่เป็นไปได้
            optimizer.lr.assign(lr.numpy())

            # ========================= writing summary data =========================
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # =============================================================== validate step ===================================================================
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # ================================================== mAP Validation =====================================================

    mAP_model = Create_Model(input_size=INPUT_RESOLUTION, CLASSES=TRAIN_CLASSES)

    # ============================================================== Start Training =======================================================
    x_epoch = []
    y_val = []

    best_val_loss = 1000  # เอาไว้เทียบค่า val loss ตอนเซฟ best model
    for epoch in range(TRAIN_EPOCHS):

        # ======================================================= Train each epoch ========================================================

        for image_data, target in trainset:  # Loop over train step (Num Batch)
            results = train_step(image_data, target) # return global_steps, optimizer.lr, giou_loss, conf_loss, prob_loss, total_loss
            cur_step = results[0] % steps_per_epoch  # เศษของ Global step หาร num batch จะได้ Current step ของ Epoch นั้นๆ

            cprint("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]), "yellow")
        
        # ====================================================== Validate valid set =======================================================

        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target) # return giou_loss, conf_loss, prob_loss, total_loss
            count += 1  # จำนวนใน validation set
            giou_val += results[0] 
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]

        # ==================================================== Tensorboard Summary =============================================================
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
        # เช็ค tensorboard ผ่านคำสั่ง tensorboard --logdir=data/ --host localhost --port 8888

        # ========================================================= End Validation ========================================================
            
        cprint("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
            format(giou_val/count, conf_val/count, prob_val/count, total_val/count), "green")

        y_val.append(total_val/count)

        # ================================================== Save Checkpoint =================================================

        if TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_loss_{:.2f}".format(total_val/count))+"_epoch_{}".format(epoch)
            model.save_weights(save_directory)
        if TRAIN_SAVE_BEST and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_loss_{:.2f}".format(total_val/count))+"_epoch_{}".format(epoch)
            model.save_weights(save_directory)
            best_val_loss = total_val/count

        # =================================================== Plot Graph =====================================================
        loss_val = total_val/count
        save_loss(loss=loss_val, epoch=epoch, model=MODEL_TYPE)

    # ======================================== Save Train progress =================================================

    tmp_loss = np.array(y_val)
    df_summit = pd.read_csv('./progress/train_progress.csv')
    df_summit[TRAIN_MODEL_NAME] = tmp_loss
    df_summit.to_csv('./progress/train_progress.csv',index=False)

# =============================================================================================================================
    try:
        mAP_model.load_weights(save_directory) # use keras weights
        get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")

# =========================================================================================================================
if __name__ == '__main__':
    main()