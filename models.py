import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ClassificationModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = FeatureExtractor()

        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, self.hyperparams["n_out"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.prediction(x)
        return out


class SiameseModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = FeatureExtractor()

        self.flatten1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.flatten2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.prediction = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.hyperparams["n_out"]),
        )

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)

        x = torch.concat([x1, x2], dim=1)
        out = self.prediction(x)
        return out.reshape(-1)
