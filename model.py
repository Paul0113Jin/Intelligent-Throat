import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.3):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)  # Apply dropout after the second convolution
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, d_in, d_model, num_blocks, d_out, dropout=0.3):
        super(ResNet, self).__init__()
        self.in_planes = d_model

        # Initial conv layer with `d_model`
        self.conv1 = nn.Conv1d(d_in, d_model, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Define each layer of the ResNet architecture
        self.layer1 = self._make_layer(d_model, d_model, num_blocks[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(d_model, d_model * 2, num_blocks[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(d_model * 2, d_model * 4, num_blocks[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(d_model * 4, d_model * 8, num_blocks[3], stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model * 8 * BasicBlock1D.expansion, d_out)

    def _make_layer(self, in_planes, planes, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock1D(in_planes, planes, stride, dropout))
            in_planes = planes * BasicBlock1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: [samples, sequence, features]
        x = x.permute(0, 2, 1) # [B, features, seq_len]

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
