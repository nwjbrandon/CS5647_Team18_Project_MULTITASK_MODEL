# import the most useful packages
import numpy as np
import matplotlib
import math
import os
from matplotlib import pyplot as plt
import IPython.display as ipd
print('finished importing')
import librosa
import librosa.display
import tensorflow as tf
import IPython.display as ipd
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
from keras.utils import to_categorical
import numpy as np
import matplotlib
import math
import os
from matplotlib import pyplot as plt
import IPython.display as ipd
print('finished importing')
# ! pip install librosa
import librosa
import librosa.display
# from google.colab import drive
# drive.mount('/content/drive')
from tqdm import tqdm

def mp3tomfcc(file_path, max_pad):
  audio, sample_rate = librosa.core.load(file_path)
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
  pad_width = max_pad - mfcc.shape[1]
  mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
  # print(mfcc.shape)
  # raise
  return mfcc


# Compile MFCCs and extract labels: https://github.com/adhishthite/sound-mnist/blob/master/utils/wav2mfcc.py
file_path = 'tone_perfect'

mfccs = []
labels = []
for f in tqdm(os.listdir(file_path)):
  if f.endswith('.mp3'):
    mfccs.append(mp3tomfcc(file_path + '/' + f, 60))
    pinyin = int(f.split("_")[0][-1])
    labels.append(pinyin)


mfccs = np.asarray(mfccs)
print(mfccs.shape)
labels = to_categorical(labels, num_classes=None)
print(labels.shape)

dim_1 = mfccs.shape[1]
dim_2 = mfccs.shape[2]
channels = 1
classes = 5

X = mfccs
print(X.shape)
X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
print(X.shape)
y = labels
input_shape = (dim_1, dim_2, channels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)