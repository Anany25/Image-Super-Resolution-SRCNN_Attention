import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2
import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)  # Output 1 channel for attention
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        attention_weights = self.sigmoid(self.conv2(out))  # Apply sigmoid to conv2 directly
        out = x * attention_weights  # Element-wise multiplication with input
        return out

class DeeperSRCNNwithAttention(nn.Module):
    def __init__(self, channels=1):
        super(DeeperSRCNNwithAttention, self).__init__()
        self.attention_block = AttentionBlock(channels, 64)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Added deeper convolutional layer
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Added deeper convolutional layer
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Added deeper convolutional layer
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Adjusted to output 1 channel
        #self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.attention_block(x)  # Apply attention block
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)  # No ReLU activation on the last layer
        #out = self.pixel_shuffle(out)
        return out
		
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

