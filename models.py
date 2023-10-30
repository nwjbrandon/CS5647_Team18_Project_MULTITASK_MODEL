import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_downsample = is_downsample
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        if self.is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.net(x) + x
        if self.is_downsample:
            x = self.downsample(x)
        return x


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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
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
            nn.Linear(512, self.hyperparams["n_classes"]),
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
            nn.Linear(512, self.hyperparams["n_tones"]),
        )

        self.pinyin_prediction = nn.Sequential(
            nn.Linear(512, self.hyperparams["n_pinyins"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        tone_out = self.tone_prediction(x)
        pinyin_out = self.pinyin_prediction(x)

        return tone_out, pinyin_out


class MultiTaskPYINClassificationModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = FeatureExtractor()

        self.fuse = nn.Sequential(
            nn.Linear(512 + 60, 512),
            nn.Sigmoid(),
        )

        self.tone_prediction = nn.Sequential(
            nn.Linear(512, self.hyperparams["n_tones"]),
        )

        self.pinyin_prediction = nn.Sequential(
            nn.Linear(512, self.hyperparams["n_pinyins"]),
        )

    def forward(self, x_mfcc, x_f0):
        x_mfcc = self.feature_extractor(x_mfcc)
        x = torch.concatenate([x_mfcc, x_f0], dim=1)

        x = self.fuse(x)

        tone_out = self.tone_prediction(x)
        pinyin_out = self.pinyin_prediction(x)

        return tone_out, pinyin_out


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = "selectitem"
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class Wav2LetterFeatureExtractor(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
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
            # nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(start_dim=1),
        )

        self.flat1 = nn.Flatten(start_dim=2)

        self.rnn = nn.Sequential(
            nn.GRU(
                4096,
                256,
                1,
                batch_first=True,
                bidirectional=True,
            ),
            SelectItem(0),
        )

        self.flat2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):
        x = self.net(x)

        x = x.permute(0, 3, 2, 1)
        x = self.flat1(x)
        x = self.rnn(x)

        x = x.permute(0, 2, 1)
        x = self.flat2(x)
        return x


class Wav2LetterModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = Wav2LetterFeatureExtractor(hyperparams)

        self.tone_prediction = nn.Sequential(
            nn.Linear(512, self.hyperparams["n_tones"]),
        )

        self.pinyin_prediction = nn.Sequential(
            nn.Linear(512, self.hyperparams["n_pinyins"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        tone_out = self.tone_prediction(x)
        pinyin_out = self.pinyin_prediction(x)

        return tone_out, pinyin_out
