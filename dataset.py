from itertools import combinations
from Levenshtein import distance
from torch.utils.data import Dataset, DataLoader
import glob
import random
import torch
from torch import nn
import torchaudio
import xml.etree.ElementTree as ET





class ToneDataset(Dataset):


	def __init__(self, mp3_folder: str, xml_folder: str, curr_tone: str) -> None:
		self.mp3_data = {}
		self.char_dict = {}
		self.all_combinations = []
		self.mp3_folder = mp3_folder
	
		same_tone = []
		for file in [filepath.split("\\")[1] for filepath in glob.glob(f"{mp3_folder}/*.mp3")]:
			tone, _ = file.split("_", 1)

			if tone == curr_tone:
				same_tone.append(file)
			else:
				self.__combinations(same_tone, xml_folder, file)
				same_tone = [file]
				curr_tone = tone

		self.__combinations(same_tone, xml_folder, file)
		self.__generate_mismatches()

		self.catergorical_len = len(bin(len(self.char_dict))[2:])
		print("[+] Finished Initialising")

		
	def __combinations(self, same_tone: list, xml_folder: str, file: str) -> None:
		"""
		Generates correct combinations from same tones
		"""
		xml_char = {}

		for correct_char, changed_char in combinations(same_tone, 2):
			
			if file in xml_char:
				char_set = xml_char[file]
			else:
				char_set = self.__parse_xml(xml_folder, file)
				xml_char[file] = char_set
			
			for character in char_set:
				self.all_combinations.append([correct_char, changed_char, character])

	def __parse_xml(self, folder: str, file: str) -> list:
		"""
		Extracts all the simplified chinese characters from the _CUSTOM xml file.
		"""
		xml_filepath = folder + "/" + file.rpartition("_")[0] + "_CUSTOM.xml"
		root = ET.parse(xml_filepath).getroot()
		chars = [character.find("simplified").text.strip() for character in root.findall("character")]

		for char in chars:
			if char not in self.char_dict:
				self.char_dict[char] = len(self.char_dict)

		return chars

	def __generate_mismatches(self) -> None:
		"""
		Creates false pairing randomly.
		"""

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

	def __getitem__(self, idx: int) -> tuple:
		correct_mp3, changed_mp3, correct_char  = self.all_combinations[idx]

		if correct_mp3 not in self.mp3_data:
			y, sr = torchaudio.load(f"{self.mp3_folder}/{correct_mp3}")
			y = torch.mean(y, 0)
			y = torchaudio.functional.resample(y, sr, 44100)
			self.mp3_data[correct_mp3] = y

		if changed_mp3 not in self.mp3_data:
			y, sr = torchaudio.load(f"{self.mp3_folder}/{changed_mp3}")
			y = torch.mean(y, 0)
			y = torchaudio.functional.resample(y, sr, 44100)
			self.mp3_data[changed_mp3] = y

		correct_tone = self.all_combinations[idx][0].split("_", 1)[0]
		changed_tone = self.all_combinations[idx][1].split("_", 1)[0]
		dissimilarity = 1 - distance(correct_tone, changed_tone) / len(correct_tone)

		# binary encoding scheme
		encoding = list(map(int, ([*bin(self.char_dict[correct_char])[2:].zfill(self.catergorical_len)])))

		print(correct_tone, changed_tone, dissimilarity, encoding)
		return self.mp3_data[correct_mp3], self.mp3_data[changed_mp3], torch.Tensor(encoding), torch.Tensor([dissimilarity])
	
	def __len__(self):
		return len(self.all_combinations)


def main():
	mp3_filepath = "./tone_perfect_all_mp3/tone_perfect"
	xml_filepath = "./tone_perfect_all_xml/tone_perfect"
	tones = ToneDataset(mp3_filepath, xml_filepath, "a1")
	obj = DataLoader(tones)

	for i, j in enumerate(obj):
		pass

if __name__ == "__main__":
	main()