import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    """
    Bottleneck block for 1D convolutional neural network.
    """
    expansion = 4  # Expands the last dimension by a factor of 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // self.expansion
        
        # Reduce dimension, process via 1D convolution, then expand again
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # If the input and output dimensions do not match, use 1x1 convolution to adjust
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class InitialConv(nn.Module):
    """
    Initial convolutional layer for processing raw audio input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1151, stride=2):
        super(InitialConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=4, layers=[2, 3, 4, 3, 3]):
        super(ResNet, self).__init__()
        self.initial_conv = InitialConv(1, 64)
        
        self.layers = nn.ModuleList()
        in_channels = 64
        out_channels = 64
        for i, num_blocks in enumerate(layers):
            stride = 2 if i > 0 else 1  # typically we increase stride every time we increase the channel size
            layer = self._make_layer(in_channels, out_channels, num_blocks, stride)
            self.layers.append(layer)
            in_channels = out_channels
            out_channels *= 2  # Doubling the number of filters with each set

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)  # Adjusting based on the last out_channels

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BottleneckBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
