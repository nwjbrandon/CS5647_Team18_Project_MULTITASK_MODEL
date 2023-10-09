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
import glob
import librosa

def train_test_split_data():
    audio_files = glob.glob("tone_perfect/*.mp3")
    train, test = train_test_split(audio_files, test_size=0.3, random_state=1)
    return train, test


class MyDataset(Dataset):
    def __init__(self, is_train):
        if is_train:
            self.x = np.load("X_train.npy")
            self.y = np.load("y_train.npy")
        else:
            self.x = np.load("X_test.npy")
            self.y = np.load("y_test.npy")


    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        inp = self.x[index]
        inp = torch.tensor(inp).permute(2, 0, 1)

        label = self.y[index]
        label = int(np.argmax(label))
        return inp, label


def create_dataloader():
    train, test = train_test_split_data()

    train_dataset = MyDataset(train)
    test_dataset = MyDataset(test)

    inp, label = train_dataset[0]
    print("Sample data: ", inp.shape, label, len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
    )
    return train_dataloader, test_dataloader

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = "selectitem"
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            # nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 120, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(120),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # nn.MaxPool2d((2, 2)),
            # nn.Conv1d(64, 64, kernel_size=13, padding=6, bias=False),
            # nn.InstanceNorm1d(64),
            # nn.MaxPool1d(2),
            # nn.Conv1d(64, 64, kernel_size=13, padding=6, bias=False),
            # nn.InstanceNorm1d(64),
            # nn.MaxPool1d(2),
            # nn.AdaptiveAvgPool2d(1),
        )

        # 5 tones
        self.prediction = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(108000, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.feature_extractor(x)
        # print(x.shape)
        # print(x.shape)
        # raise
        x = self.prediction(x)
        # print(x.shape)
        # raise
        return x


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

    optimizer = Adadelta(
        model.parameters(), 
        lr=0.001,     
        rho=0.95,
        eps=1e-07,
        weight_decay=0,
    )
    # scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.9, verbose=False,)
    criterion = nn.CrossEntropyLoss()

    device = "mps"
    model = Model()
    out = model(inp)
    model.to(device)
    print(model)


    n_epochs = 50
    for epoch in range(n_epochs):
        train_loss = train(
            model, train_dataloader, optimizer, criterion, device
        )
        valid_loss = validate(
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
        log = "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
            epoch + 1, n_epochs, np.round(train_loss, 10), np.round(valid_loss, 10),
        )
        print(log)
        with open("ckpts/loss.txt", "a") as f:
            f.write(f"{log}\n")

if __name__ == "__main__":
    main()