from itertools import combinations
# from Levenshtein import distance
from torch.utils.data import Dataset, DataLoader
import glob
import random
import torch
from torch import nn
import torchaudio
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from pytorch_metric_learning import losses
import numpy as np

class CreateDataset():
    def __init__(self, mp3_folder: str, xml_folder: str) -> None:
        self.mp3_folder = mp3_folder
        self.xml_folder = xml_folder
        self.char_dict = {}
        self.all_combinations = []

    def __combine_same_tones(self, curr_tone):
        same_tone = []
        all_mp3s = glob.glob(f"{self.mp3_folder}/*.mp3")
        filepaths =  [filepath.split("\\")[1] for filepath in all_mp3s]

        for file in filepaths:
            tone, _ = file.split("_", 1)

            if tone == curr_tone:
                same_tone.append(file)

            else:
                self.__combinations(same_tone, file)
                same_tone = [file]
                curr_tone = tone

        self.__combinations(same_tone, file)

    def __combinations(self, same_tone: list, file: str) -> None:
        xml_char = {}

        for correct_char, changed_char in combinations(same_tone, 2):
            
            if file in xml_char:
                char_set = xml_char[file]
            else:
                char_set = self.__parse_xml(file)
                xml_char[file] = char_set
            
            for character in char_set:
                self.all_combinations.append([correct_char, changed_char, character])

    def __parse_xml(self, file: str) -> list:
        xml_filepath = self.xml_folder + "/" + file.rpartition("_")[0] + "_CUSTOM.xml"
        root = ET.parse(xml_filepath).getroot()
        chars = [character.find("simplified").text.strip() for character in root.findall("character")]

        for char in chars:
            if char not in self.char_dict:
                self.char_dict[char] = len(self.char_dict)

        return chars

    def __generate_mismatches(self) -> None:
        max_range = len(self.all_combinations)
        tone_of = lambda idx : self.all_combinations[idx][0].split("_", 1)[0]
        
        for idx in range(max_range):
            
            while True:
                rand_idx = random.randint(0, max_range)
                
                first_tone = tone_of(idx)
                second_tone = tone_of(rand_idx)

                if first_tone != second_tone:
                    correct_mp3 = self.all_combinations[idx][0]
                    wrong_mp3 = self.all_combinations[rand_idx][0]
                    correct_char = self.all_combinations[idx][2]
                    self.all_combinations.append([correct_mp3, wrong_mp3, correct_char])
                    break

    def create(self, init_tone="a1") -> tuple:
        self.__combine_same_tones(init_tone)
        self.__generate_mismatches()
        random.shuffle(self.all_combinations)
    
        return self.all_combinations, len(bin(len(self.char_dict))[2:]), self.char_dict

class ToneDataset(Dataset):
    def __init__(self, dataset: list, mp3_folder: str, char_dict: dict, enc_len: int) -> None:
        self.mp3_folder = mp3_folder
        self.dataset = dataset
        self.char_dict = char_dict
        self.enc_len = enc_len
        self.mp3_data = {}

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __process_mp3(self, filepath: str) -> torch.float64:
        if filepath not in self.mp3_data:
            y, sr = torchaudio.load(f"{self.mp3_folder}/{filepath}")
            y = torch.mean(y, 0)
            y = torchaudio.functional.resample(y, sr, 44100)

            """
            TODO
            Get MFCC/Mel Spectrogram and Tone and standardise frame length
            """

            ############## NOT FINAL ################
            y = y[:400]
            ############## NOT FINAL ################
            self.mp3_data[filepath] = y

        return self.mp3_data[filepath]
    
    def __dist_pinyin(self, correct_mp3: str, changed_mp3: str) -> float:
        t1 = correct_mp3.split("_", 1)[0]
        t2 = changed_mp3.split("_", 1)[0]
        
        return 1 if t1 == t2 else 0
        # return 1 - distance(t2, t1) / len(t1)

    def __getitem__(self, idx) -> tuple:
        correct_mp3, changed_mp3, correct_char = self.dataset[idx]

        correct_waveform = self.__process_mp3(correct_mp3)
        changed_waveform = self.__process_mp3(changed_mp3)
        similarity_score = self.__dist_pinyin(correct_mp3, changed_mp3)
        encoding = list(map(int, ([*bin(self.char_dict[correct_char])[2:].zfill(self.enc_len)])))

        return correct_waveform, changed_waveform, torch.Tensor(encoding), torch.Tensor([similarity_score])

class Model(nn.Module):
    def __init__(self, in_features=400, out_features=12):
        """
        PLACEHOLDER
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, 250)
        self.fc2 = nn.Linear(250, 400)
        self.pred_head = nn.Linear(400, out_features)
        self.softmax = nn.Softmax()

    def __siamese_twin(self, x):
        """ PLACEHOLDER
        TODO
        write the classification part here
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        correct_embedding = self.__siamese_twin(x[0])
        changed_embedding = self.__siamese_twin(x[1])

        """
        change prediction head
        """
        preds = self.pred_head(correct_embedding)
        preds = self.softmax(preds)

        return correct_embedding, changed_embedding, preds

def trainer(dataloader, model, epoch=5):
    loss_func_1 = losses.ContrastiveLoss()
    loss_func_2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    alpha = 0.7

    for i in range(epoch):
        print(f"[+] Epoch: {i}")

        for data in tqdm(dataloader):
            truth, changed, labels, similarity = data

            truth, changed, preds = model([truth, changed])

            optimizer.zero_grad()
            cross_entropy_loss = loss_func_2(preds, labels)
            contastive_loss = loss_func_1(changed, torch.squeeze(similarity), ref_emb=truth)

            # Try Random Weighting: https://openreview.net/forum?id=jjtFD8A1Wx
            # Refs: https://github.com/SamsungLabs/MTL/tree/master
            # Refs: https://github.com/median-research-group/LibMTL
            total_loss = alpha * cross_entropy_loss + (1 - alpha) * contastive_loss
            total_loss.backward()

            optimizer.step()

def train_test_val_split(dataset, train_split=0.7, val_split=0.2):
    train_split = int(train_split * len(dataset))
    val_split = train_split + int(val_split * len(dataset))

    train = dataset[:train_split]
    val = dataset[train_split:val_split]
    test = dataset[val_split:]

    return train, val, test

if __name__ == "__main__":
    mp3_filepath = "./tone_perfect_all_mp3/tone_perfect"
    xml_filepath = "./tone_perfect_all_xml/tone_perfect"

    dataset, bin_encoding_len, char_dict = CreateDataset(mp3_filepath, xml_filepath).create()
    train, val, test = train_test_val_split(dataset)

    """
    Using test dataset for faster testing purposes. NOT FINAL.
    """
    test = ToneDataset(test, mp3_filepath, char_dict, bin_encoding_len)
    test = DataLoader(test, batch_size=64, shuffle=True)

    model = Model()
    trainer(test, model)