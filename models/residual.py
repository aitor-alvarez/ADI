from torch import nn
from torchaudio import transforms
from torch.functional import F

#Residual blocks combined with BLTSM

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out



class ResidualLSTM(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResidualLSTM, self).__init__()
        self.in_channels=128
        self.speclayer = transforms.MelSpectrogram()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(1, self.in_channels, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = self.make_layer(block, 128, layers[0])
        self.lstm = nn.LSTM(2193, 1000, batch_first=True, bidirectional=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256000, 5)


    def make_layer(self, block, out_channels, blocks, kernel=3, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = x.unsqueeze(1)
        in_spec = self.speclayer(x)
        out = self.initial_layer(in_spec)
        out_residual = self.layer2(out)
        batch, time = out_residual.size()[:2]
        out = out_residual.reshape(batch, time, -1)
        lstm_out, hidden = self.lstm(out)
        in_ffn = self.flatten(lstm_out)
        print(in_ffn.shape)
        output= self.fc(in_ffn)
        return F.log_softmax(output, dim=1)