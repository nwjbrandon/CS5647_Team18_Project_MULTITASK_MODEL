import collections
import glob
import json

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor


class TonePerfectDataset(Dataset):
    def __init__(self, audio_fnames, hyperparams):
        super().__init__()
        self.audio_fnames = audio_fnames
        self.hyperparams = hyperparams
        self.sampling_rate = hyperparams["sampling_rate"]
        self.n_mfcc = hyperparams["n_mfcc"]
        self.n_mels = hyperparams["n_mels"]
        self.max_pad = self.hyperparams["max_pad"]
        self.dataset = self.hyperparams["dataset"]
        self.max_length = self.hyperparams["max_length"]
        self.preprocess_type = self.hyperparams["preprocess_type"]

        self.w2v = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

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
        inp = self.load_inp(audio_fname)

        if self.dataset == "TONES":
            label = self.get_tone_label(audio_fname)
        elif self.dataset == "PINYINS":
            label = self.get_pinyin_label(audio_fname)
        elif self.dataset == "LABELS":
            label = self.get_label(audio_fname)
        else:
            raise "Invalid Dataset"

        return inp, label

    def load_inp(self, audio_fname):
        if self.preprocess_type == "mfcc":
            inp = self.load_mfcc(audio_fname)
        elif self.preprocess_type == "melspectrogram":
            inp = self.load_melspectrogram(audio_fname)
        elif self.preprocess_type == "raw":
            inp = self.load_waveform(audio_fname)
        else:
            raise "Invalid Dataset"
        return inp

    def load_mfcc(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        pad_width = self.max_pad - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        mfcc = torch.tensor(mfcc).unsqueeze(0)
        return mfcc

    def load_melspectrogram(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname)
        melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=self.n_mels)
        pad_width = self.max_pad - melspectrogram.shape[1]
        melspectrogram = np.pad(melspectrogram, pad_width=((0, 0), (0, pad_width)), mode="constant")
        melspectrogram = torch.tensor(melspectrogram).unsqueeze(0)
        return melspectrogram

    def load_waveform(self, audio_fname):
        audio, sample_rate = librosa.core.load(audio_fname, sr=self.w2v.sampling_rate)
        waveform = self.w2v(
            audio,
            sampling_rate=self.w2v.sampling_rate,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        waveform = waveform["input_values"].squeeze(0)
        n_pad = self.max_length - len(waveform)
        waveform = F.pad(waveform, (0, n_pad), "constant", 0)
        return waveform

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


def train_test_split_by_tones(audio_files, hyperparams):
    train, test = [], []
    hash_map = collections.defaultdict(list)
    for audio_file in audio_files:
        gt = audio_file.split("/")[-1].split("_")[0][-1]
        hash_map[gt].append(audio_file)
    for key in hash_map:
        for audio_file in hash_map[key]:
            if "MV3" in audio_file or "FV3" in audio_file:
                test.append(audio_file)
            else:
                train.extend(audio_file)
    return train, test


def train_test_split_by_pinyins(audio_files, hyperparams):
    train, test = [], []
    hash_map = collections.defaultdict(list)
    for audio_file in audio_files:
        gt = audio_file.split("/")[-1].split("_")[0][:-1]
        hash_map[gt].append(audio_file)
    for key in hash_map:
        for audio_file in hash_map[key]:
            if "MV3" in audio_file or "FV3" in audio_file:
                test.append(audio_file)
            else:
                train.extend(audio_file)
    return train, test


def train_test_split_by_labels(audio_files, hyperparams):
    train, test = [], []
    hash_map = collections.defaultdict(list)
    for audio_file in audio_files:
        gt = audio_file.split("/")[-1].split("_")[0]
        hash_map[gt].append(audio_file)
    for key in hash_map:
        for audio_file in hash_map[key]:
            if "MV3" in audio_file or "FV3" in audio_file:
                test.append(audio_file)
            else:
                train.extend(audio_file)
    return train, test


def train_test_split_data(hyperparams):
    dataset = hyperparams["dataset"]
    train_test_split_data_map = {
        "TONES": train_test_split_by_tones,
        "PINYINS": train_test_split_by_pinyins,
        "LABELS": train_test_split_by_labels,
        "MULTITASK": train_test_split_by_pinyins,
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
        inp = self.load_inp(audio_fname)

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
