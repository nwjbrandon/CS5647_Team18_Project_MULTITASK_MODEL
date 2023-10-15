import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, out, tgt):
        loss = self.criterion(out, tgt)
        return loss
