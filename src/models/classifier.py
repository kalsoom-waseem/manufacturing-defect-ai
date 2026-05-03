import torch.nn as nn
from torchvision import models


class DefectClassifier(nn.Module):

    def __init__(self, num_classes=6, pretrained=True, freeze_layers=True):
        super(DefectClassifier, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Freeze first
        if freeze_layers:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        # Unfreeze layer4
        for name, param in self.model.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True

        # Replace final layer — always trainable
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def get_gradcam_target_layer(self):
        return self.model.layer4[-1]

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)