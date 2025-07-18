
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights

def get_resnet50(name='resnet50', pretrained=False):
    # Choose weights based on the `pretrained` flag
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    
    # Load the ResNet model with or without pretrained weights
    resnet = getattr(models, name)(weights=weights)
    
    # Remove classification head (use as feature extractor)
    resnet.fc = nn.Identity()
    
    return resnet

def get_resnet18(name='resnet18', pretrained=False):
    # Choose weights based on the `pretrained` flag
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    
    # Load the ResNet model with or without pretrained weights
    resnet = getattr(models, name)(weights=weights)
    
    # Remove classification head (use as feature extractor)
    resnet.fc = nn.Identity()
    
    return resnet
