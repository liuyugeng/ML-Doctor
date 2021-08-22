import torch
import torch.nn as nn
import torch.nn.functional as F

class attrinf_attack_model(nn.Module):
    def __init__(self, inputs, outputs):
        super(attrinf_attack_model, self).__init__()
        self.classifier = nn.Linear(inputs, outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x