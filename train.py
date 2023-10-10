import glob

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, audio_fnames, hyperparams):
        self.audio_fnames = audio_fnames
        self.hyperparams = hyperparams
        self.n_mfcc = hyperparams["n_mfcc"]
        self.max_pad = self.hyperparams["max_pad"]

    def __len__(self):
        return len(self.audio_fnames)

    def __getitem__(self, index):
        audio_fname = self.audio_fnames[index]
        inp = self.preprocess_data(audio_fname)
        label = self.preprocess_label(audio_fname)
        inp = torch.tensor(inp).unsqueeze(0)
        return inp, label

    def preprocess_data(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        pad_width = self.max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        return mfcc

    def preprocess_label(self, audio_file):
        tone = audio_file.split("/")[-1].split("_")[0][-1]
        tone = int(tone) - 1
        assert tone <= 3
        return tone


def train_test_split_data(hyperparams):
    audio_files = glob.glob("tone_perfect/*.mp3")
    train, test = train_test_split(audio_files, test_size=hyperparams["test_size"], random_state=hyperparams["random_state"])
    return train, test


def create_dataloader(hyperparams):
    train_data, test_data = train_test_split_data(hyperparams)

    train_dataset = MyDataset(train_data, hyperparams)
    test_dataset = MyDataset(test_data, hyperparams)

    inp, label = train_dataset[0]
    print("Sample data: ", inp.shape, label, len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
    )
    return train_dataloader, test_dataloader


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

        # 4 tones
        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(64, self.hyperparams["n_tones"]),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.prediction(x)
        return out


def train(model, dataloader, optimizer, criterion, metric, device):
    model.train()
    losses = []
    y_true = []
    y_pred = []
    for img, label in tqdm(dataloader):
        inp = img.float().to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(out.data, 1)
        label = label.detach().cpu().numpy().tolist()
        y_true.extend(label)
        predicted = predicted.detach().cpu().numpy().tolist()
        y_pred.extend(predicted)
        losses.append(loss.item())

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    loss = np.sum(losses) / len(losses)
    acc = metric(y_pred, y_true)
    return acc, loss


def validate(model, dataloader, optimizer, criterion, metric, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            inp = img.float().to(device)
            label = label.to(device)
            out = model(inp)
            loss = criterion(out, label)

            _, predicted = torch.max(out.data, 1)
            label = label.detach().cpu().numpy().tolist()
            y_true.extend(label)
            predicted = predicted.detach().cpu().numpy().tolist()
            y_pred.extend(predicted)
            losses.append(loss.item())

    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    loss = np.sum(losses) / len(losses)
    acc = metric(y_pred, y_true)
    return acc, loss


def save_checkpoint(epoch, model, optimizer, train_loss, valid_loss, train_acc, valid_acc):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
        },
        f"ckpts/pytorch_model_{epoch}.pth",
    )


def print_training_log(epoch, n_epochs, train_loss, valid_loss, train_acc, valid_acc):
    log = "Epoch: {}/{}, Train Acc={}, Val Acc={}, Train Loss={}, Val Loss={}".format(
        epoch + 1,
        n_epochs,
        np.round(train_acc, 4),
        np.round(valid_acc, 4),
        np.round(train_loss, 4),
        np.round(valid_loss, 4),
    )
    print(log)


def main():
    hyperparams = {
        "batch_size": 64,
        "n_tones": 4,
        "learning_rate": 0.01,
        "test_size": 0.3,
        "random_state": 1,
        "n_epochs": 20,
        "n_mfcc": 128,
        "max_pad": 60,
    }

    train_dataloader, test_dataloader = create_dataloader(hyperparams)
    inp, label = next(iter(train_dataloader))
    print("Sample Batch: ", inp.shape, label.shape)

    model = Model(hyperparams)
    out = model(inp)
    print("Prediction: ", out.shape)
    device = "mps"
    model.to(device)
    print(model)

    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"])
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=1,
        gamma=0.9,
        verbose=False,
    )
    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=hyperparams["n_tones"])

    n_epochs = hyperparams["n_epochs"]
    records = []
    for epoch in range(n_epochs):
        train_acc, train_loss = train(model, train_dataloader, optimizer, criterion, metric, device)
        valid_acc, valid_loss = validate(model, test_dataloader, optimizer, criterion, metric, device)
        scheduler.step()

        save_checkpoint(epoch, model, optimizer, train_loss, valid_loss, train_acc, valid_acc)
        print_training_log(epoch, n_epochs, train_loss, valid_loss, train_acc, valid_acc)
        records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv("ckpts/training_logs.csv", sep=",", index=False)


if __name__ == "__main__":
    main()
