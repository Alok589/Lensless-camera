import os
import shutil


DATASET_DIR = '/thesis/'
GT_DIR = os.path.join(DATASET_DIR, 'groundtruth')
# MEAS_DIR_1 = os.path.join(DATASET_DIR, 'Meas1')
# MEAS_DIR_2 = os.path.join(DATASET_DIR, 'Meas2')
# MEAS_DIR_3 = os.path.join(DATASET_DIR, 'Meas3')
# MEAS_DIR_4 = os.path.join(DATASET_DIR, 'Meas4')
# MEAS_DIR_5 = os.path.join(DATASET_DIR, 'Meas5')
# MEAS_DIR_6 = os.path.join(DATASET_DIR, 'Meas6')
# MEAS_DIR_7 = os.path.join(DATASET_DIR, 'Meas7')
# MEAS_DIR_8 = os.path.join(DATASET_DIR, 'Meas8')

target_dir = "/thesis/measurements/"
src_folders = ["Meas2", "Meas3", "Meas4", "Meas5", "Meas6", "Meas7", "Meas8"]

for src_folder in src_folders:
    meas_folder_path = os.path.join(DATASET_DIR, src_folder)
    for src_meas_folder in os.listdir(meas_folder_path):
        # print(file_folder)
        if src_meas_folder in os.listdir(target_dir):
            print(src_meas_folder)
            target_meas_folder_path = os.path.join(
                DATASET_DIR, src_folder, src_meas_folder)
            for meas_file in os.listdir(target_meas_folder_path):

                src_path = os.path.join(target_meas_folder_path, meas_file)
                print(src_path)
                target_path = os.path.join(
                    target_dir, src_meas_folder, meas_file)
                print(target_path)
                shutil.move(src_path, target_path)
        else:
            # if not os.path.exists(directory):
            print("___________Folder created____________")
            os.makedirs(os.path.join(target_dir, src_meas_folder))
            target_meas_folder_path = os.path.join(
                DATASET_DIR, src_folder, src_meas_folder)
            for meas_file in os.listdir(target_meas_folder_path):

                src_path = os.path.join(target_meas_folder_path, meas_file)
                print(src_path)
                target_path = os.path.join(
                    target_dir, src_meas_folder, meas_file)
                print(target_path)
