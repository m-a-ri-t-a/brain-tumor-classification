import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights
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


class MRI_VGG16_Classifier(nn.Module):
    def __init__(self, num_classes, train_vgg16=False):
        super(MRI_VGG16_Classifier, self).__init__()

        # Load pre-trained VGG16 model
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.vgg16.parameters():
            param.requires_grad = train_vgg16

        in_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Pass the input through the VGG16 network
        x = self.vgg16(x)

        # Apply softmax to get probability distribution
        x = F.softmax(x, dim=1)
        return x