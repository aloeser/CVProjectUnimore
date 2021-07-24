import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, inplanes, planes, stride):
        super(ResidualBlock, self).__init__()
        self._f = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        if (stride > 1) or (inplanes != planes):
            self._g = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self._g = nn.Identity()

    def forward(self, x):
        return F.relu(self._f(x) + self._g(x), True)


class MOREsNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 8, 
                 load_pretrained:str = None, model_state_name: str = None):
        super(MOREsNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1),
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes, bias=True)
        )

        if load_pretrained and model_state_name:
            checkpoint = torch.load(load_pretrained)
            model_state_name = model_state_name if model_state_name else 'model_state_dict'
            self.load_state_dict(checkpoint[model_state_name])

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x
