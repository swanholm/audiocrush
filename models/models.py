import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ConvModelSimple4(nn.Module):
    def __init__(self, num_classes=3, kernel_sizes=[351, 51, 31, 11, 5, 3, 3], strides=[4, 4, 2, 2, 1, 1, 1]):
        super(ConvModelSimple4, self).__init__()
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

class ConvModelSimple5(nn.Module):
    def __init__(self, num_classes=3, kernel_sizes=[551, 71, 51, 31, 15, 5, 3, 3], strides=[4, 2, 1, 1, 1, 1, 1, 1]):
        super(ConvModelSimple5, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_sizes[0], stride=strides[0], dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_sizes[1], stride=strides[1], dilation=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_sizes[2], stride=strides[2], dilation=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel_sizes[6], stride=strides[6])
        self.conv8 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=kernel_sizes[7], stride=strides[7])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# class SincLstmModel1(nn.Module):

#     SINCNET_DEFAULTS = {"stride": 10}
#     LSTM_DEFAULTS = {
#         "hidden_size": 128,
#         "num_layers": 2,
#         "bidirectional": True,
#         "monolithic": True,
#         "dropout": 0.0,
#     }
#     LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

#     def __init__(
#         self,
#         sincnet: Optional[dict] = None,
#         lstm: Optional[dict] = None,
#         linear: Optional[dict] = None,
#         sample_rate: int = 16000,
#         num_channels: int = 1
#     ):
#         super().__init__()

#         sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
#         sincnet["sample_rate"] = sample_rate
#         lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
#         lstm["batch_first"] = True
#         linear = merge_dict(self.LINEAR_DEFAULTS, linear)

#         self.sincnet = SincNet(**self.hparams.sincnet)

#         monolithic = lstm["monolithic"]
#         if monolithic:
#             multi_layer_lstm = dict(lstm)
#             del multi_layer_lstm["monolithic"]
#             self.lstm = nn.LSTM(60, **multi_layer_lstm)

#         else:
#             num_layers = lstm["num_layers"]
#             if num_layers > 1:
#                 self.dropout = nn.Dropout(p=lstm["dropout"])

#             one_layer_lstm = dict(lstm)
#             one_layer_lstm["num_layers"] = 1
#             one_layer_lstm["dropout"] = 0.0
#             del one_layer_lstm["monolithic"]

#             self.lstm = nn.ModuleList(
#                 [
#                     nn.LSTM(
#                         60
#                         if i == 0
#                         else lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1),
#                         **one_layer_lstm
#                     )
#                     for i in range(num_layers)
#                 ]
#             )

#         if linear["num_layers"] < 1:
#             return

#         lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
#             2 if self.hparams.lstm["bidirectional"] else 1
#         )
#         self.linear = nn.ModuleList(
#             [
#                 nn.Linear(in_features, out_features)
#                 for in_features, out_features in pairwise(
#                     [
#                         lstm_out_features,
#                     ]
#                     + [self.hparams.linear["hidden_size"]]
#                     * self.hparams.linear["num_layers"]
#                 )
#             ]
#         )

#     def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
#         """Pass forward

#         Parameters
#         ----------
#         waveforms : (batch, channel, sample)

#         Returns
#         -------
#         scores : (batch, frame, classes)
#         """

#         outputs = self.sincnet(waveforms)

#         if self.hparams.lstm["monolithic"]:
#             outputs, _ = self.lstm(
#                 rearrange(outputs, "batch feature frame -> batch frame feature")
#             )
#         else:
#             outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
#             for i, lstm in enumerate(self.lstm):
#                 outputs, _ = lstm(outputs)
#                 if i + 1 < self.hparams.lstm["num_layers"]:
#                     outputs = self.dropout(outputs)

#         if self.hparams.linear["num_layers"] > 0:
#             for linear in self.linear:
#                 outputs = F.leaky_relu(linear(outputs))

#         return self.activation(self.classifier(outputs))

