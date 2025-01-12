from enum import Enum
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
import torch.nn.functional as F

RADIMAGENET_RESNET = "pretrained_weights/ResNet50.pt"
RADIMAGENET_DENSENET = "pretrained_weights/DenseNet121.pt"

class Weights(Enum):
    IMN = 'imagenet'
    RIMN = 'radimagenet'

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes, dropout):
        super().__init__()
        self.drop_out = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        return x
    
class ResnetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            base_model = resnet50(pretrained=False)

        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])  
                 
    def forward(self, x):
        return self.backbone(x)
    
class MRIResNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout, weights: Weights, train_resnet=False):
        super(MRIResNetClassifier, self).__init__()

        if weights not in [Weights.IMN, Weights.RIMN]:
            raise ValueError("Invalid weights, choose between imagenet and radimagenet.")  
        
        backbone = ResnetBackbone(pretrained=(weights==Weights.IMN))
        if weights == Weights.RIMN:
            backbone.load_state_dict(torch.load(RADIMAGENET_RESNET, weights_only=True))

        if not train_resnet:
            for param in backbone.parameters():
                param.requires_grad = False
            
        classifier = Classifier(2048, num_classes, dropout)
        self.seq = nn.Sequential(backbone, classifier)

    def forward(self, x):
        x = self.seq(x)
        x = F.softmax(x, dim=1)
        return x


class DenseNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            base_model = densenet121(pretrained=False)

        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:-1])  
                 
    def forward(self, x):
        return self.backbone(x)
    
class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout, weights: Weights, train_densenet=False):
        super(DenseNetClassifier, self).__init__()

        if weights not in [Weights.IMN, Weights.RIMN]:
            raise ValueError("Invalid weights, choose between imagenet and radimagenet.")  
        
        backbone = DenseNetBackbone(pretrained=(weights==Weights.IMN))
        if weights == Weights.RIMN:
            backbone.load_state_dict(torch.load(RADIMAGENET_DENSENET, weights_only=True))

        if not train_densenet:
            for param in backbone.parameters():
                param.requires_grad = False
            
        classifier = Classifier(1024*7*7, num_classes, dropout)
        self.seq = nn.Sequential(backbone, classifier)

    def forward(self, x):
        x = self.seq(x)
        x = F.softmax(x, dim=1)
        return x
