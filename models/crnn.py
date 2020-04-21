import torch.nn as nn
from torchaudio import transforms
from torch.functional import F


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        #self.speclayer = transforms.MFCC(n_mfcc=128)
        self.speclayer = transforms.MelSpectrogram()
        #self.speclayer = transforms.MelScale()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(0.1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Dropout(0.1)
        )
        self.GRU = nn.GRU(6272, 256, 2, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10496, 5)


    def forward(self, x):
        x = x.unsqueeze(1)
        in_spec = self.speclayer(x)
        out = self.layer1(in_spec)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.transpose(1, -1)

        # out -> (batch, time, channel*freq)
        batch, time = out.size()[:2]
        out = out.reshape(batch, time, -1)
        gru_out, hidden = self.GRU(out)
        in_dense = self.flatten(gru_out)
        out_dense = self.fc1(in_dense)
        return F.log_softmax(out_dense, dim=1)
