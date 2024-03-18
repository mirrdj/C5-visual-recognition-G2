from typing import Dict
import torch.nn as nn
import torch
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class BaseNet(nn.Module):
    def __init__(self, params):
        super(BaseNet, self).__init__()

        if params['COCO']=='False':
            model = models.densenet121(pretrained=True)

            # Freeze all parameters
            # for param in model.parameters():
            #     param.requires_grad = False

            num_features = model.classifier.in_features
            print(f"Number of features: {num_features}")
            model.classifier = nn.Linear(num_features, params['output']) #Add layer with 8 classes
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=True).backbone
            print(model)
            print(f"Number of features: {model.out_channels}")

        # for child in list(model.children())[-params['unfroze']:]:
        #     for param in child.parameters():
        #         param.requires_grad = True

        self.model = model

    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def forward(self, x):
        x = self.model(x)
        return x


    def get_features(self, x):
        pass # TODO: Implement this

class BaseCOCO(nn.Module):
    def __init__(self, params):
        super(BaseNet, self).__init__()
    # Model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
        model = fasterrcnn_resnet50_fpn(weights='COCO_V1').backbone
        num_features = model.classifier.in_features
        print(f"Number of features: {num_features}")
        model.classifier = nn.Linear(num_features, params['output']) #Add layer with 8 classes

        self.model = model
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def forward(self, x):
        x = self.model(x)
        return x
