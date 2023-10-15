import collections
import glob
import json

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class TonePerfectDataset(Dataset):
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
        return self.binarise_gt(gt)

    def binarise_gt(self, gt):
        n_classes_bin = bin(self.n_classes)[2:]  # "Ignore prefix 0b"
        gt_bin = bin(gt)[2:]

        res = [0] * len(n_classes_bin)
        j = len(n_classes_bin) - 1
        for i in range(len(gt_bin) - 1, -1, -1):
            c = gt_bin[i]
            if c == "1":
                res[j] = 1
            j -= 1
        return torch.tensor(res).float()


def train_test_split_data(hyperparams):
    dataset = hyperparams["dataset"]

    audio_files = glob.glob("tone_perfect/*.mp3")
    train, test = [], []
    if dataset == "TONES":
        train, test = train_test_split(
            audio_files, test_size=hyperparams["test_size"], random_state=hyperparams["random_state"]
        )
    elif dataset == "PINYINS":
        hash_map = collections.defaultdict(list)
        for audio_file in audio_files:
            gt = audio_file.split("/")[-1].split("_")[0][:-1]
            hash_map[gt].append(audio_file)
        for key in hash_map:
            group = hash_map[key]
            train.extend(group[:-1])
            test.append(group[-1])
    elif dataset == "LABELS":
        hash_map = collections.defaultdict(list)
        for audio_file in audio_files:
            gt = audio_file.split("/")[-1].split("_")[0]
            hash_map[gt].append(audio_file)
        for key in hash_map:
            group = hash_map[key]
            train.extend(group[:-1])
            test.append(group[-1])
    else:
        raise "Invalid Dataset"

    return train, test


def create_dataloader_tone_perfect(hyperparams):
    train_data, test_data = train_test_split_data(hyperparams)

    train_dataset = TonePerfectDataset(train_data, hyperparams)
    test_dataset = TonePerfectDataset(test_data, hyperparams)

    print("n train: ", len(train_dataset))
    print("n test: ", len(test_dataset))

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


class TonePerfectSiameseDataset(Dataset):
    def __init__(self, df, hyperparams):
        self.df = df
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
        return len(self.df)

    def __getitem__(self, index):
        audio_fname1 = self.df.iloc[index]["audio_fname1"]
        audio_fname2 = self.df.iloc[index]["audio_fname2"]
        is_same = self.df.iloc[index]["is_same"]

        inp1 = self.preprocess_data(audio_fname1)
        inp1 = torch.tensor(inp1).unsqueeze(0)

        inp2 = self.preprocess_data(audio_fname2)
        inp2 = torch.tensor(inp2).unsqueeze(0)

        return inp1, inp2, is_same

    def preprocess_data(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        pad_width = self.max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        return mfcc


def create_dataloader_tone_perfect_siamese(hyperparams):
    train_df = pd.read_csv("annotation_train.csv").sample(frac=1, random_state=42), 
    test_df = pd.read_csv("annotation_test.csv").sample(frac=1, random_state=42)

    train_dataset = TonePerfectSiameseDataset(train_df, hyperparams)
    test_dataset = TonePerfectSiameseDataset(test_df, hyperparams)

    print("n train: ", len(train_dataset))
    print("n test: ", len(test_dataset))

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


def create_dataloaders(hyperparams, mode=None):
    if mode == "SIAMESE":
        return create_dataloader_tone_perfect_siamese(hyperparams)
    else:
        return create_dataloader_tone_perfect(hyperparams)
