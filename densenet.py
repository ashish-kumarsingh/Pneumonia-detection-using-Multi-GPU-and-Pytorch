import os
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import train_test_split
from PIL import Image
# from tensorflow.python.client import device_lib
import subprocess


def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


# print(device_lib.list_local_devices())

CPU_COUNT = multiprocessing.cpu_count()
GPU_COUNT = len(get_gpu_name())
print("Available CPUs: ", CPU_COUNT)
print("Available GPUs: ", GPU_COUNT)

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import os

data_dir = 'Images'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)


## Training with 1 GPU




trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=8)

print('Train_Data: ', len(train_data))
print('Test_Data: ', len(test_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

device = 'cuda:0'

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device)

print('\n\n******************* Training the model with 1 GPU *******************\n\n')

import time

traininglosses = []
testinglosses = []
testaccuracy = []
totalsteps = []
epochs = 3
steps = 0
running_loss = 0
print_every = 5
epoch_start = time.time()

print('Training')

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            traininglosses.append(running_loss / print_every)
            testinglosses.append(test_loss / len(testloader))
            testaccuracy.append(accuracy / len(testloader))
            totalsteps.append(steps)
            print(f"Device {device}.."
                  f"Epoch {epoch + 1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()

elapse_time_1 = time.time() - epoch_start
print("Training time  with 1 GPU {}".format(elapse_time_1))

#checkpoint = {
 #   'parameters': model.parameters,
#    'state_dict': model.state_dict()
#}

#torch.save(checkpoint, 'models/densenet.pth')



print('\n\n******************* Training the model with 2 GPUs *******************\n\n')


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True,num_workers=8)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,num_workers=8)

print('Train_Data: ', len(train_data))
print('Test_Data: ', len(test_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)


print('Inside Multi-GPU')
model = nn.DataParallel(model,device_ids=[0,1])

model.to(device)

import time

traininglosses = []
testinglosses = []
testaccuracy = []
totalsteps = []
epochs = 3
steps = 0
running_loss = 0
print_every = 5
epoch_start = time.time()

print('Training')

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            traininglosses.append(running_loss / print_every)
            testinglosses.append(test_loss / len(testloader))
            testaccuracy.append(accuracy / len(testloader))
            totalsteps.append(steps)
            print(f"Device {device}.."
                  f"Epoch {epoch + 1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()

elapse_time_2 = time.time() - epoch_start
print("Training time {}".format(elapse_time_2))


print('\n\n******************* Training the model with 4 GPUs *******************\n\n')


trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,num_workers=8)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128,num_workers=8)

print('Train_Data: ', len(train_data))
print('Test_Data: ', len(test_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)


print('Inside Multi-GPU')
model = nn.DataParallel(model,device_ids=[0,1,2,3])

model.to(device)

import time

traininglosses = []
testinglosses = []
testaccuracy = []
totalsteps = []
epochs = 3
steps = 0
running_loss = 0
print_every = 5
epoch_start = time.time()

print('Training')

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            traininglosses.append(running_loss / print_every)
            testinglosses.append(test_loss / len(testloader))
            testaccuracy.append(accuracy / len(testloader))
            totalsteps.append(steps)
            print(f"Device {device}.."
                  f"Epoch {epoch + 1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()

elapse_time_4 = time.time() - epoch_start

print("Training time with 1 GPU {}".format(elapse_time_1))
print("Training time with 2 GPUs{}".format(elapse_time_2))
print("Training time with 4 GPUs{}".format(elapse_time_4))









