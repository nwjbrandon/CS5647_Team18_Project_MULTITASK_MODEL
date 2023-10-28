import torch.nn as nn
from transformers import AutoModelForAudioClassification


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


class MultiTaskClassificationModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = FeatureExtractor()

        self.tone_prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, self.hyperparams["n_tones"]),
        )

        self.pinyin_prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, self.hyperparams["n_pinyins"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        tone_out = self.tone_prediction(x)
        pinyin_out = self.pinyin_prediction(x)

        return tone_out, pinyin_out


class W2VModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        pretrained_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-2])

        self.tone_prediction = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(768, self.hyperparams["n_tones"]),
        )

        self.pinyin_prediction = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(768, self.hyperparams["n_pinyins"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x).last_hidden_state
        x = x.permute(0, 2, 1)

        tone_out = self.tone_prediction(x)
        pinyin_out = self.pinyin_prediction(x)

        return tone_out, pinyin_out
