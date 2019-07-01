import torch.nn as nn 


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(1))
        layers.append(nn.ReLU(inplace=True))
        self.baseline_cnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.baseline_cnn(x)
        return out