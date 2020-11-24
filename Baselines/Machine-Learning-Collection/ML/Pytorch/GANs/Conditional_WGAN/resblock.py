
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append('./DNA_project/Baselines/Machine-Learning-Collection/ML/Pytorch/GANs/Conditional_WGAN/')


from utils_dna_cwgan import gradient_penalty, save_checkpoint, load_checkpoint
import os
from model_dna_cwgan import Discriminator_1D, Generator_1D, initialize_weights,Dna_FeatureExtractor
from utils_dna_cwgan import ToTensor, DnaHotEncoding, StrandDataset,Encoded_to_DNA,Test_environment_setup
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


dummy = torch.ones((1, 1, 1, 1))
