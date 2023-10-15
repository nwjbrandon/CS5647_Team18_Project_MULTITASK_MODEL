import glob
import json

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, audio_fnames, hyperparams):
        self.audio_fnames = audio_fnames
        self.hyperparams = hyperparams
        self.n_mfcc = hyperparams["n_mfcc"]
        self.max_pad = self.hyperparams["max_pad"]
        self.dataset = self.hyperparams["dataset"]
        self.n_classes = self.hyperparams["n_classes"]

        self.tones = []
        self.pinyins = []
        self.labels = []
        self.load_labels()

    def load_labels(self):
        with open("tones.json") as f:
            self.tones = json.load(f)
        with open("pinyins.json") as f:
            self.pinyins = json.load(f)
        with open("labels.json") as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.audio_fnames)

    def __getitem__(self, index):
        audio_fname = self.audio_fnames[index]
        inp = self.preprocess_data(audio_fname)
        inp = torch.tensor(inp).unsqueeze(0)

        if self.dataset == "TONES":
            label = self.get_tone_label(audio_fname)
        elif self.dataset == "PINYINS":
            label = self.get_pinyin_label(audio_fname)
        elif self.dataset == "LABELS":
            label = self.get_label(audio_fname)
        else:
            raise "Invalid Dataset"

        return inp, label

    def preprocess_data(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        pad_width = self.max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        return mfcc

    def get_tone_label(self, audio_file):
        gt = audio_file.split("/")[-1].split("_")[0][-1]
        gt = int(gt)
        gt = self.tones.index(gt)
        assert gt != -1
        return gt

    def get_pinyin_label(self, audio_file):
        gt = audio_file.split("/")[-1].split("_")[0][:-1]
        gt = self.pinyins.index(gt)
        assert gt != -1
        return gt

    def get_label(self, audio_file):
        gt = audio_file.split("/")[-1].split("_")[0]
        gt = self.labels.index(gt)
        assert gt != -1
        return gt


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
