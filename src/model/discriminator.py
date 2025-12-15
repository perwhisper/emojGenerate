import torch
import torch.nn as nn
import torch.nn.functional as F

class EmojiDiscriminator(nn.Module):
    def __init__(self, img_dim=3, au_dim=16, identity_feat_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(img_dim, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.real_head = nn.Conv2d(1024, 1, 4, 1, 0)
        self.fc_au = nn.Linear(1024, au_dim)
        self.fc_identity = nn.Linear(1024, identity_feat_dim)

    def _forward_conv(self, img):
        x = self.leaky_relu(self.conv1(img))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.bn(self.conv3(x)))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))
        return x

    def forward(self, img):
        x = self._forward_conv(img)
        real_logit = self.real_head(x)
        pooled = self.avgpool(x).view(x.size(0), -1)
        pred_au = self.fc_au(pooled)
        pred_identity = self.fc_identity(pooled)
        return real_logit, pred_au, pred_identity
