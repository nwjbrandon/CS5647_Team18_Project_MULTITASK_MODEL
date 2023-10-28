import collections
import glob
import json

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class TonePerfectDataset(Dataset):
    def __init__(self, audio_fnames, hyperparams):
        super().__init__()
        self.audio_fnames = audio_fnames
        self.hyperparams = hyperparams
        self.n_mfcc = hyperparams["n_mfcc"]
        self.max_pad = self.hyperparams["max_pad"]
        self.dataset = self.hyperparams["dataset"]

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


def train_test_split_tones_data(audio_files, hyperparams):
    train, test = train_test_split(audio_files, test_size=hyperparams["test_size"], random_state=hyperparams["random_state"])
    return train, test


def train_test_split_pinyins_data(audio_files, hyperparams):
    train, test = [], []
    hash_map = collections.defaultdict(list)
    for audio_file in audio_files:
        gt = audio_file.split("/")[-1].split("_")[0][:-1]
        hash_map[gt].append(audio_file)
    for key in hash_map:
        group = hash_map[key]
        train.extend(group[:-1])
        test.append(group[-1])
    return train, test


def train_test_split_tones_pinyins_data(audio_files, hyperparams):
    train, test = [], []
    hash_map = collections.defaultdict(list)
    for audio_file in audio_files:
        gt = audio_file.split("/")[-1].split("_")[0]
        hash_map[gt].append(audio_file)
    for key in hash_map:
        group = hash_map[key]
        train.extend(group[:-1])
        test.append(group[-1])
    return train, test


def train_test_split_data(hyperparams):
    dataset = hyperparams["dataset"]
    train_test_split_data_map = {
        "TONES": train_test_split_tones_data,
        "PINYINS": train_test_split_tones_pinyins_data,
        "LABELS": train_test_split_tones_pinyins_data,
        "MULTITASK": train_test_split_tones_pinyins_data,
    }

    audio_files = glob.glob("tone_perfect/*.mp3")
    train, test = train_test_split_data_map[dataset](audio_files, hyperparams)
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


class TonePerfectMultiTaskDataset(TonePerfectDataset):
    def __init__(self, audio_fnames, hyperparams):
        super().__init__(audio_fnames, hyperparams)

    def __getitem__(self, index):
        audio_fname = self.audio_fnames[index]
        inp = self.preprocess_data(audio_fname)
        inp = torch.tensor(inp).unsqueeze(0)

        tone = self.get_tone_label(audio_fname)
        pinyin = self.get_pinyin_label(audio_fname)

        return inp, tone, pinyin


def create_dataloader_tone_perfect_multitask(hyperparams):
    audio_files = glob.glob("tone_perfect/*.mp3")
    train_data, test_data = train_test_split(
        audio_files, test_size=hyperparams["test_size"], random_state=hyperparams["random_state"]
    )

    train_dataset = TonePerfectMultiTaskDataset(train_data, hyperparams)
    test_dataset = TonePerfectMultiTaskDataset(test_data, hyperparams)

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
    if mode == "MULTITASK":
        return create_dataloader_tone_perfect_multitask(hyperparams)
    else:
        return create_dataloader_tone_perfect(hyperparams)
