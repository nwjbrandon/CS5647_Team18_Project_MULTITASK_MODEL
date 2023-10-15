import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, out, tgt):
        loss = self.criterion(out, tgt)
        return loss


class MultiTaskClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tone_criterion = nn.CrossEntropyLoss()
        self.pinyin_criterion = nn.CrossEntropyLoss()

    def forward(self, tone_out, tone_tgt, pinyin_out, pinyin_tgt):
        tone_loss = self.tone_criterion(tone_out, tone_tgt)
        pinyin_loss = self.pinyin_criterion(pinyin_out, pinyin_tgt)
        return tone_loss + pinyin_loss


class SiameseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, out, tgt):
        loss = self.criterion(out, tgt)
        return loss
