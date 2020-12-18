import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch.optim as optim


def __init__(self, image_paths, targets, resize=None, augmentations=None):
    # """
    # :param image_paths: list of path to images
    # :param targets: numpy array
    # :param resize: tuple, e.g. (256, 256), resizes image if not None
    # :param augmentations: albumentation augmentations
    # """
    self.image_paths = image_paths
    self.targets = targets
    self.resize = resize


def __len__(self):

    return len(self.image_paths)


def __getitem__(self, item):

    image = Image.open(self.image_paths[item])
    image = image.convert("RGB")
    targets = self.targets[item]


def train(data_loader, model, optimizer, device):
    model.train()
    for data in data_loader:
        inputs = data["image"]
        targets = data["targets"]
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.MSELoss()
        loss.backward()


# model = Unet()
# learning_rate = 0.001
# n_iters = 100
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(modle.parameters(), lr = learning_rate, eps = n_iters,)

# for epoc in range (n_iters):
#     y_pred = forward(image)
#     l = criterion(image, y_pred)
