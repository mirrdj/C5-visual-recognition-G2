import torch.nn as nn
from torchvision import models

class TripletNet(nn.Module):
    def __init__(self, base_net):
        super(TripletNet, self).__init__()
        self.base_net = base_net

    def forward(self, x1, x2, x3):
        output1 = self.base_net(x1)
        output2 = self.base_net(x2)
        output3 = self.base_net(x3)

        return output1, output2, output3
    
    def state_dict(self):
        return self.base_net.state_dict()
    
    def load_state_dict(self, load):
        return self.base_net.load_state_dict(load)