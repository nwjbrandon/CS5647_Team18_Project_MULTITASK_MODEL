import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, Adam, Adadelta
from torch.optim.lr_scheduler import StepLR
from collections import deque
from tqdm import tqdm
from torchvision.models import resnet18
import glob
import librosa

def train_test_split_data():
    audio_files = glob.glob("tone_perfect/*.mp3")
    train, test = train_test_split(audio_files, test_size=0.2, random_state=42)
    return train, test

train_data, test_data = train_test_split_data()

class MyDataset(Dataset):
    def __init__(self, is_train):
        if is_train:
            self.audio_fnames = train_data
        else:
            self.audio_fnames = test_data

    def __len__(self):
        return len(self.audio_fnames)

    def __getitem__(self, index):
        audio_fname = self.audio_fnames[index]
        inp = self.preprocess_data(audio_fname)
        label = self.preprocess_label(audio_fname)
        inp = torch.tensor(inp).unsqueeze(0)
        return inp, label

    def preprocess_data(self, audio_fname, max_pad=60):
        audio, sample_rate = librosa.core.load(audio_fname)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
        pad_width = max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfcc

    def preprocess_label(self, audio_file):
        tone = audio_file.split("/")[-1].split("_")[0][-1]
        tone = int(tone) - 1
        return tone

def create_dataloader():
    train_dataset = MyDataset(True)
    test_dataset = MyDataset(False)

    inp, label = train_dataset[0]
    print("Sample data: ", inp.shape, label, len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=20,
        shuffle=True,
    )
    return train_dataloader, test_dataloader

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # # 5 tones
        self.prediction = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28800, 128),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.Dropout(0.4),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.prediction(x)
        return out


def train(model, dataloader, optimizer, criterion, device):
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

    loss = np.sum(losses) / len(losses)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc, loss


def validate(model, dataloader, optimizer, criterion, device):
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

    loss = np.sum(losses) / len(losses)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc, loss


def main():
    train_dataloader, test_dataloader = create_dataloader()
    inp, label = next(iter(train_dataloader))
    print("Sample Batch: ", inp.shape, label.shape)

    model = Model()
    out = model(inp)
    print("Prediction: ", out.shape)

    optimizer = Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.9, verbose=False,)
    criterion = nn.CrossEntropyLoss()

    device = "mps"
    model = Model()
    out = model(inp)
    model.to(device)
    print(model)


    n_epochs = 50
    for epoch in range(n_epochs):
        train_acc, train_loss = train(
            model, train_dataloader, optimizer, criterion, device
        )
        valid_acc, valid_loss = validate(
            model, test_dataloader, optimizer, criterion, device
        )
        # scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            f"ckpts/pytorch_model_{epoch}.pth",
        )
        log = "Epoch: {}/{}, Train Acc={}, Val Acc={}".format(
            epoch + 1, n_epochs, np.round(train_acc, 10), np.round(valid_acc, 10),
        )
        print(log)
        with open("ckpts/loss.txt", "a") as f:
            f.write(f"{log}\n")

if __name__ == "__main__":
    main()