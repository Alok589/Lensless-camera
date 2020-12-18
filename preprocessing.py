import cv2
import matplotlib.pyplot as pyplot
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as transforms
import os
import random
import numpy as np

# img_folder = 'E:\\WorkSpace\\Lensless\\dataset\\measurements\\n01440764'
# for i in range(11):
#     file = random.choice(os.listdir(img_folder))
#     image_path = os.path.join(img_folder, file)
#     img = mpimg.imread(image_path)
#     ax = plt.subplot(1, 11, i+1)
#     ax.title.set_text(file)
#     plt.imshow(img)

# plt.show()


def create_dataset(img_folder):

    img_data_array = []
    class_name = []
    count = 0
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):

            image_path = os.path.join(img_folder, dir1,  file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (572, 572), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            #image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
        count += 1
        if count == 10:
            break

    return img_data_array, class_name


# extract the image array and class name
img_data, class_name = create_dataset(
    'E:\\WorkSpace\\Lensless\\dataset\\measurements')

print("")
#img = cv2.imread(datadir)
