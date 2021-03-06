import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

import math

# Set these to whatever you want for your gaussian filter
kernel_size = 15
sigma = 1

channels = 3
# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
x_cord = torch.arange(kernel_size)
x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
y_grid = x_grid.t()
xy_grid = torch.stack([x_grid, y_grid], dim=-1)

mean = (kernel_size - 1) / 2.
variance = torch.tensor(sigma ** 2., dtype=torch.float)

# Calculate the 2-dimensional gaussian kernel which is
# the product of two gaussian distributions for two different
# variables (in this case called x and y)
gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                  torch.exp(
                      (-torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                       (2 * variance)).float()
                  )
# Make sure sum of values in gaussian kernel equals 1.
gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

# Reshape to 2d depthwise convolutional weight
gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, bias=False)

gaussian_filter.weight.data = gaussian_kernel
gaussian_filter.weight.requires_grad = False


def low_pass(img, filter):
    return filter(img)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Lambda(lambda x: TF.adjust_contrast(x, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: low_pass(x, gaussian_filter)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dirs = [r'../input/imagenet16', r'../input/stylizedimagenet16']
data_dir = r'../input/imagenet16'
# data_dir = r'../input/stylizedimagenet16'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def top5(outputs, labels):
    # print(outputs.shape)
    # print(labels.shape)
    _, idx = outputs.topk(5, 1)
    return (idx == labels.unsqueeze(1)).sum().sum()


def eval_model(model, criterion):
    model.eval()
    phase = 'val'
    with torch.no_grad():
        running_top5acc = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            top5acc = criterion(outputs, labels)
            # print(top5acc)
            # return

            # statistics
            running_top5acc += top5acc.item()
            running_corrects += torch.sum(preds == labels.data)

            # print(running_top5acc)

        epoch_loss = running_top5acc / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))


if __name__ == '__main__':
    model = models.resnet50(pretrained=False)
    #     for param in model.parameters():
    #         param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 16)
    model.load_state_dict(torch.load('../input/in16-res50-200/resnet50-IN.pth'))
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = top5

    eval_model(model, criterion)
