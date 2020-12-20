import os
import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from Dense_Unet import Dense_Unet
from PIL import Image
import cv2
import engine


if __name__ == "__main__":

    project_path = '/thesis/'
    data_path = "/thesis/dataset/"
    file_name = "file_name.txt"

    # cuda/cpu device
    device = "cuda:7"
    # let's train for 10 epochs
    epochs = 10
    #model = Dense_Unet
    # load the dataframe
    # df = pd.read_csv(os.path.join(data_path, "train.csv"))

    # load the files
    with open(file_name, "r") as f:
        file_names = f.read()[:-1].split('\n')

    meas_data = [os.path.join(data_path, "measurements", file.split('_')[
                              0], file+'..png') for file in file_names]
    gt_data = [os.path.join(data_path, "groundtruth", file.split('_')[
                            0], file+'.JPEG') for file in file_names]

    train_X, test_X, train_Y, test_Y = train_test_split(
        meas_data, gt_data, test_size=0.25)

    train_X, val_X, train_Y, val_Y = train_test_split(
        train_X, train_Y, test_size=0.15)

    # # binary targets numpy array
    # targets = df.target.values
    # # fetch out model, we will try both pretrained
    # # and non-pretrained weights
    model = Dense_Unet()
    # move model to device
    model.to(device)

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_X[:30],
        targets=train_Y[:30],
        resize=(256, 256)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)

    valid_dataset = dataset.ClassificationDataset(
        image_paths=val_X,
        targets=val_Y,
        resize=(256, 256)
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    test_dataset = dataset.ClassificationDataset(
        image_paths=test_X,
        targets=test_Y,
        resize=(256, 256)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1
    )
    data = train_dataset[23]

    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, verbose=True)
    # train and print auc score for all epochs^

    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
            valid_loader, model, device=device)
    # wrap model and optimizer with NVIDIA's apex
    # this is used for mixed precision training
    # if you have a GPU that supports mixed precision,
    # this is very helpful as it will allow us to fit larger images
    # and larger batches
#     model, optimizer=amp.initialize(
#     model, optimizer, opt_level="O1", verbosity=0
#     )
# if we have more than one GPU, we can use both of them!
# if torch.cuda.device_count() > 1:
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model)
# some logging
    # print(f"Training batch size: {TRAINING_BATCH_SIZE}")
    # print(f"Test batch size: {TEST_BATCH_SIZE}")
    # print(f"Epochs: {EPOCHS}")
    # print(f"Image size: {IMAGE_SIZE}")
    # print(f"Number of training images: {len(train_dataset)}")
    # print(f"Number of validation images: {len(valid_dataset)}")
    # print(f"Encoder: {ENCODER}")

# # loop over all epochs
# for epoch in range(EPOCHS):
#     print(f"Training Epoch: {epoch}")
#     # train for one epoch
#     train(
#     train_dataset,
#     train_loader,
#     model,
#     criterion,
#     optimizer
#     )

# print(f"Validation Epoch: {epoch}")
# # calculate validation loss
# val_log=evaluate(
# valid_dataset,
# valid_loader,
# model
# )
# # step the scheduler
# scheduler.step(val_log["loss"])
# print("\n")
