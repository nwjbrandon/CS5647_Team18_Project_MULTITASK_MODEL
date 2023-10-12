import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(64, self.hyperparams["n_tones"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.prediction(x)
        return x, out


class PreTrainedModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.hyperparams = hyperparams
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


if __name__ == "__main__":
    hyperparams = {
        "batch_size": 64,
        "n_tones": 4,
        "learning_rate": 0.01,
        "test_size": 0.3,
        "random_state": 1,
        "n_epochs": 10,
        "n_mfcc": 128,
        "max_pad": 60,
    }
    with torch.no_grad():
        model = Model(hyperparams)
        model.eval()
        weights = torch.load("ckpts/pytorch_model_9.pth")["model_state_dict"]
        model.load_state_dict(weights)

        preTrainedModel = PreTrainedModel(hyperparams)
        preTrainedModel.feature_extractor = model.feature_extractor

        inp = torch.rand(2, 1, 128, 60)
        embeddings1, out = model(inp)
        embeddings2 = preTrainedModel(inp)
        print("Embeddings 1:")
        print(embeddings1[0, 0, 0, :5])
        print("Embeddings 2:")
        print(embeddings2[0, 0, 0, :5])
        assert torch.allclose(embeddings1, embeddings2)
