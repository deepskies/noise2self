import torch.nn as nn
import torch.nn.functional as F

'''
@author Michael Lepori
@date 8/26/19

DnFCN8 architecture

Hybrid of FCN8 and DnCNN architectures. 
This architecture follows the standard FCN8 architecture at first, shrinking the
image substantially in order to get a coarse representation of the image. In order to
allow the final image to incorporate the very fine grained information necessary for
denoising operations, an additional skip layer is added immediately after the first 2 
convolutions. This skip layer is at full resolution, 128 by 128 pixels.

These skip layers are then transformed into 128 by 128 scale, and concatenated as 
if they were different channels. This is in contrast to the traditional FCN8 
architecture, where the skip layers are added together elementwise.

This four channel image is then fed into the "final convolution" layers, which mimic
a 10 layer DnCNN. 

The idea behind this architecture is that a DnCNN may perform better if it has
direct access to information at various levels of coarseness. 

It may be necessary to increase the depth of the DnCNN portion of the network in order
to achieve top results.
'''

class DnFCN8(nn.Module):
    def __init__(self, in_channels=1):
        super(DnFCN8, self).__init__()

        kernel = 3
        padding = 1

        self.pool = nn.MaxPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel, padding=padding)
        self.conv2 = nn.Conv2d(64, 64, kernel, padding=padding)

        self.skip_fc1 = nn.Conv2d(64, 1, 1)

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel, padding=padding)
        self.conv4 = nn.Conv2d(128, 128, kernel, padding=padding)

        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel, padding=padding)
        self.conv6 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.conv7 = nn.Conv2d(256, 256, kernel, padding=padding)

        self.skip_fc2 = nn.Conv2d(256, 32, 1)

        self.bn4 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel, padding=padding)
        self.conv9 = nn.Conv2d(512, 512, kernel, padding=padding)
        self.conv10 = nn.Conv2d(512, 512, kernel, padding=padding)

        self.skip_fc3 = nn.Conv2d(512, 64, 1)

        self.bn5 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, kernel, padding=padding)
        self.conv12 = nn.Conv2d(512, 512, kernel, padding=padding)
        self.conv13 = nn.Conv2d(512, 512, kernel, padding=padding)

        self.bn6 = nn.BatchNorm2d(512)

        self.fc14 = nn.Conv2d(512, 4096, 1)
        self.fc15 = nn.Conv2d(4096, 4096, 1)
        self.fc16 = nn.Conv2d(4096, 128, 1)

        self.upsample32x = nn.ConvTranspose2d(128, 1, kernel_size=32, stride=32)
        self.upsample16x = nn.ConvTranspose2d(64, 1, kernel_size=16, stride=16)
        self.upsample8x = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=8)

        self.final_conv1 = nn.Conv2d(4, 64, kernel, padding=padding)
        self.bn7 = nn.BatchNorm2d(64)
        
        self.final_conv2 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn8 = nn.BatchNorm2d(64)
        
        self.final_conv3 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn9 = nn.BatchNorm2d(64)
        
        self.final_conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn10 = nn.BatchNorm2d(64)
        
        self.final_conv5 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn11 = nn.BatchNorm2d(64)
        
        self.final_conv6 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn12 = nn.BatchNorm2d(64)
        
        self.final_conv7 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn13 = nn.BatchNorm2d(64)
        
        self.final_conv8 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn14 = nn.BatchNorm2d(64)
        
        self.final_conv9 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.bn15 = nn.BatchNorm2d(64)
        
        self.final_conv10 = nn.Conv2d(64, 1, kernel, padding=padding)


    def forward(self, x):

        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        out1 = self.skip_fc1(x)

        x = self.pool(x)

        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool(x)

        x = self.bn3(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = self.pool(x)

        out2 = self.skip_fc2(x)
        out2 = self.upsample8x(out2)

        x = self.bn4(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = self.pool(x)

        out3 = self.skip_fc3(x)
        out3 = self.upsample16x(out3)

        x = self.bn5(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))      

        x = self.pool(x)

        x = self.bn6(x)
        x = F.relu(self.fc14(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc16(x))  

        out4 = self.upsample32x(x)
        
        outs = torch.cat([out1, out2, out3, out4], dim=1)
        
        outs = F.relu(self.final_conv1(outs))
        
        outs = self.bn7(outs)
        outs = F.relu(self.final_conv2(outs))
        
        outs = self.bn8(outs)
        outs = F.relu(self.final_conv3(outs))
        
        outs = self.bn9(outs)
        outs = F.relu(self.final_conv4(outs))
        
        outs = self.bn10(outs)
        outs = F.relu(self.final_conv5(outs))
        
        outs = self.bn11(outs)
        outs = F.relu(self.final_conv6(outs))
        
        outs = self.bn12(outs)
        outs = F.relu(self.final_conv7(outs))
        
        outs = self.bn13(outs)
        outs = F.relu(self.final_conv8(outs))
        
        outs = self.bn14(outs)
        outs = F.relu(self.final_conv9(outs))
        
        outs = self.bn15(outs)
        out = self.final_conv10(outs)
        
        return out