# model.py
import torch
from torch import nn
from torchvision import models
import os

def load_checkpoint(filepath, arch='vgg16'):
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    if arch not in ['vgg16', 'vgg19']:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = getattr(models, arch)(pretrained=True)

    checkpoint = torch.load(filepath, map_location='cpu')
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    # Ajouter le chargement de class_to_idx
    class_to_idx = checkpoint.get('class_to_idx', None)


    return model,class_to_idx
