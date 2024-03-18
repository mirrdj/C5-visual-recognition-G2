import torch.nn as nn
from torchvision import models


class SiameseNet(nn.Module):
    def __init__(self, base_net):
        super(SiameseNet, self).__init__()
        self.base_net = base_net

    def forward(self, x1, x2):
        output1 = self.base_net(x1)
        output2 = self.base_net(x2)
        return output1, output2
    
    def state_dict(self):
        return self.base_net.state_dict()

    def load_state_dict(self, load):
        return self.base_net.load_state_dict(load)
    
    def get_features(self, x):
        self.base_net.get_features(x)