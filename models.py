import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class MRIResNetClassifier(nn.Module):
    def __init__(self, num_classes, train_resnet = False):
        super(MRIResNetClassifier, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        for param in self.resnet.parameters():
            param.requires_grad = train_resnet
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.softmax(x, dim=1)
        return x
