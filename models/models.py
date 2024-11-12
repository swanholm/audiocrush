import torch
import torch.nn as nn
import torch.nn.functional as F

# some toy-example models to play around with
class ConvModelSimple(nn.Module):
    def __init__(self, num_classes=4, kernel_sizes=[33, 5, 5, 3], strides=[16, 8, 4, 2]):
        super(ConvModelSimple, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_sizes[0], stride=strides[0], dilation=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class MultiScaleDetectModel(nn.Module): # TODO!!!
    def __init__(self, num_classes=4):
        super(MultiScaleDetectModel, self).__init__()
        convs_1 = []
        for i in range(5):
            convs_1.append(nn.Conv1d(
                in_channels=1, 
                out_channels=1, 
                kernel_size = 51, 
                stride= 16 / i,
                dilation= 2 ** i
            ))

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ConvModelSimple2(nn.Module):
    def __init__(self, num_classes=4, kernel_sizes=[201, 21, 5, 5, 3, 3], strides=[8, 4, 2, 2, 1, 1]):
        super(ConvModelSimple2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_sizes[0], stride=strides[0], dilation=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1], dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_sizes[5], stride=strides[5])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ConvModelSimple2Regu(nn.Module):
    def __init__(self, num_classes=4, kernel_sizes=[201, 21, 5, 5, 3, 3], strides=[8, 4, 2, 2, 1, 1], dropout_rate=0.3):
        super(ConvModelSimple, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_sizes[0], stride=strides[0], dilation=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1], dilation=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(dropout_rate)  # Add dropout starting here
        
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_sizes[4], stride=strides[4])
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(dropout_rate)
        
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_sizes[5], stride=strides[5])
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(dropout_rate)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        x = self.drop6(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ConvModelSimple3(nn.Module):
    def __init__(self, num_classes=3, kernel_sizes=[251, 31, 11, 5, 3, 3, 3], strides=[4, 4, 2, 2, 1, 1, 1]):
        super(ConvModelSimple3, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_sizes[0], stride=strides[0], dilation=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1], dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_sizes[6], stride=strides[6])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ConvModelSimple3regu(nn.Module):
    def __init__(self, num_classes=3, kernel_sizes=[551, 331, 51, 35, 23, 13, 9], strides=[3, 3, 2, 2, 1, 1, 1]):
        super(ConvModelSimple3regu, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_sizes[0], stride=strides[0], dilation=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[1], stride=strides[1], dilation=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=kernel_sizes[2], stride=strides[2], dilation=1)
        self.bn3 = nn.BatchNorm1d(96)
        self.drop3 = nn.Dropout(0.1)

        self.conv4 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(0.2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_sizes[4], stride=strides[4])
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.3)

        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_sizes[5], stride=strides[5])
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(0.4)

        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_sizes[6], stride=strides[6])
        self.bn7 = nn.BatchNorm1d(1024)
        self.drop7 = nn.Dropout(0.5)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        x = self.drop6(F.relu(self.bn6(self.conv6(x))))
        x = self.drop7(F.relu(self.bn7(self.conv7(x))))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
