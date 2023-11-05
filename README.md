# CS55647 Project

## Install
- Install python dependencies
```
conda env create -f environment.yml
```

## Dataset
- Place the dataset of mp3 files inside the folder `tone_perfect/`


## Train
- Train Multitask Model
```
python3 train_multitask.py
```
- Train RNN Model
```
python3 train_w2c.py
```
- Train Multitask Model with PYIN
```
python3 train_pinyin.py
```
- Train Model For Tone or PinYin only
```
python3 train.py
```


## Results
- Refers to `Results.ipynb` for the visualisation of the training logs
