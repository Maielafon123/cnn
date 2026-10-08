import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0
            )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.linear3 = nn.Linear(256, 10)



    def forward(self,x):
        x_out = F.relu(self.conv1(x))
        x_out = F.relu(self.conv2(x_out))
        x_out = F.max_pool2d(x_out, kernel_size=2, stride=2)
        x_out = F.relu(self.conv3(x_out))
        x_out = F.relu(self.conv4(x_out))
        x_out = torch.flatten(x_out, 1)
        x_out = F.relu(self.linear1(x_out))
        x_out = F.relu(self.linear2(x_out))
        x_out = self.linear3(x_out)
        return x_out
