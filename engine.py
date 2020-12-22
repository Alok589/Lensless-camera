 
# engine.py
import torch
import torch.nn as nn
from tqdm import tqdm
# import model
from torch._C import device
from torch.optim import optimizer
import numpy as np

def train(data_loader, model, optimizer, device):

    model.train()
    batch_MSEs = []
    for data in tqdm(data_loader):
        # remember, we have image and targets
        # in our dataset class
        inputs = data["image"]
        targets = data["targets"]
        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward step of model
        outputs = model(inputs)
        # calculate loss
        loss = nn.MSELoss()(outputs, targets)
        batch_MSEs.append(loss.item())
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        #print(loss.item())
    batch_MSEs = np.array(batch_MSEs)        
    epoch_loss = np.mean(batch_MSEs)
    print(epoch_loss)
    return epoch_loss    
        

def evaluate(data_loader, model, device):

    # """
    # This function does evaluation for one epoch
    # :param data_loader: this is the pytorch dataloader
    # :param model: pytorch model
    # :param device: cuda/cpu
    # """

    # put model in evaluation mode
    print("_____________validation_____________")
    model.eval()

    # init lists to store targets and outputs
    batch_MSEs = []
    # we use no_grad context
    with torch.no_grad():
        for idx, data in enumerate(data_loader, 1):
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # do the forward step to generate prediction
            output = model(inputs)

            # convert targets and outputs to lists
            batch_mse = ((output-targets)**2).mean().item()
            #print("batch"+str(idx) + " loss:" ,batch_mse)

            batch_MSEs.append(batch_mse)
            # return final output and final targets
    batch_MSEs = np.array(batch_MSEs)        
    epoch_loss = np.mean(batch_MSEs)
    print(epoch_loss)
    return epoch_loss
