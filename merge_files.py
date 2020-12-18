import os
import json
import numpy as np
import shutil

DATASET_DIR = '/home/chauhan/thesis/Dataset/'
GT_DIR = os.path.join(DATASET_DIR, 'ground_truth')
MEAS_DIR_1 = os.path.join(DATASET_DIR, 'mes_pt_1')
MEAS_DIR_2 = os.path.join(DATASET_DIR, 'mes_pt_2')

print(GT_DIR)
print(MEAS_DIR_1)
print(MEAS_DIR_2)


for dir in os.listdir(MEAS_DIR_2):
    print("folder name", dir)
    for file in os.listdir(os.path.join(MEAS_DIR_2, dir)):
        print("    " + file)
        src = os.path.join(MEAS_DIR_2, dir, file)
        dest = os.path.join(MEAS_DIR_1, dir, file)
        shutil.move(src, dest)
