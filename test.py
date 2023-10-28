import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

labels = [1, 2, 3, 4]
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
num_labels = len(id2label)

audio, sample_rate = librosa.core.load("tone_perfect/a1_FV1_MP3.mp3")
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
pad_width = 60 - mfcc.shape[1]
mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
plt.imshow(mfcc)
plt.show()

audio, sample_rate = librosa.core.load("tone_perfect/a1_FV1_MP3.mp3")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
inputs = feature_extractor(
    audio, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True, return_tensors="pt"
)
print(inputs["input_values"][0].shape, feature_extractor.sampling_rate, audio.shape, sample_rate)


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pretrained_model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
        )
        pretrained_model.gradient_checkpointing_enable()

        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.net = nn.Linear(768, 4)

    def forward(self, x):
        x = self.feature_extractor(x).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.net(x)
        return x


model = Model()
out = model(inputs["input_values"])
# out = model(**inputs)
print(out.shape)
