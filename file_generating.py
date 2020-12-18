import os
import numpy as np

DATASET_DIR = '/thesis/dataset/'
GT_DIR = os.path.join(DATASET_DIR, 'groundtruth')
MEAS_DIR = os.path.join(DATASET_DIR, 'measurements')

print(GT_DIR)
print(MEAS_DIR)

meas_fnames_list = []
gt_fnames_list = []

for dir in os.listdir(GT_DIR):
    folder_name = os.path.join(GT_DIR, dir)
    if os.path.isdir(folder_name):
        for file in os.listdir(folder_name):
            # print("     file name", file.split('.')[0])
            gt_fnames_list.append(file.split('.')[0])

#print("gt count", len(gt_fnames_list))

for dir in os.listdir(MEAS_DIR):
    folder_name = os.path.join(MEAS_DIR, dir)
    if os.path.isdir(folder_name):
        for file in os.listdir(folder_name):
            # print("     file name", file.split('.')[0])
            meas_fnames_list.append(file.split('.')[0])

meas_fnames_list = np.array(meas_fnames_list)
gt_fnames_list = np.array(gt_fnames_list)

with open("file_name.txt", "w") as f:
    for file in meas_fnames_list:
        f.write(file + '\n')
