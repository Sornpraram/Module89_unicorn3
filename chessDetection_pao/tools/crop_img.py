import cv2
import os

i = 0

read_dir = "C:\\Module89_unicorn3\\chessDetection_pao\\dataset\\uncrop\\TEST"
for filename in os.listdir(read_dir):
    save_dir = "C:\\Module89_unicorn3\\chessDetection_pao\\dataset\\crop\\Test"
    i += 1

    image = cv2.imread(read_dir + "\\" + filename)
    print(read_dir + "\\" + filename)
    crop_img = image[0:1080,430:1510]
    # save_dir = save_dir + "pawn" #acd[1]
    # print(save_dir + "\\" + filename)
    cv2.imwrite(save_dir + "\\" + str(i) + ".jpg", crop_img)
