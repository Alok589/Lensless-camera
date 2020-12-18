import os
import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import dataset
import engine
from Dense_Unet import Dense_Unet


if __name__ == "__main__":

    data_path = "/thesis/dataset/"
# cuda/cpu device
device = "cuda"
# let's train for 10 epochs
epochs = 10
# load the dataframe
df = pd.read_csv(os.path.join(data_path, "train.csv"))
# fetch all image ids
images = df.ImageId.values.tolist()

images = [
    os.path.join(data_path, "train_png", i + ".png") for i in images
]
# binary targets numpy array
targets = df.target.values
# fetch out model, we will try both pretrained
# and non-pretrained weights
model = Dense_Unet
# move model to device
model.to(device)

train_images, valid_images, train_targets, valid_targets = train_test_split(
    images, targets, stratify=targets, random_state=42
)
# fetch the ClassificationDataset class
train_dataset = dataset.ClassificationDataset(
    image_paths=train_images,
    targets=train_targets,
    resize=(227, 227))
# torch dataloader creates batches of data
# from classification dataset class
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4
)
# same for validation data
valid_dataset = dataset.ClassificationDataset(image_paths=valid_images,
                                              targets=valid_targets,
                                              resize=(227, 227),
                                              augmentations=aug,
                                              )
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=4
)
# simple Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
# train and print auc score for all epochs
for epoch in range(epochs):
    engine.train(train_loader, model, optimizer, device=device)
    predictions, valid_targets = engine.evaluate(
        valid_loader, model, device=device)
