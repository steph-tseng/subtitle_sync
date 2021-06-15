# %%
from logging import error
import pathlib
from pathlib import Path
from pytube import YouTube
import os
import sys
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = 'DATA'
if os.path.isdir(DATA_DIR) != True:
  os.mkdir(DATA_DIR)

yt = YouTube('https://www.youtube.com/watch?v=OfrMqoyOJVE')
t = yt.streams.filter(only_audio=True)
t[0].download('DATA')

# %%
caption = yt.captions['en.ehkg1hFWq8A']
print(caption)

f = open(f"{os.path.join(DATA_DIR, yt.title)}.srt", "w")
f.write(caption.generate_srt_captions())

# %%

## Standalone boilerplate before relative imports
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from .audio_converter import *

train_file = 'Reuben Poutine Is Our Greasy Love Letter to Canada'

sound = generateSingleDataset(train_file, cut_data=False, len_mfcc=LEN_MFCC, step_mfcc=STEP_MFCC, hop_len=HOP_LEN, freq=FREQ)


######################################

# %%
subs = pysrt.open(os.path.join(DATA_DIR, train_file) + '.srt', encoding='iso-8859-1')
# %%
from pathlib import Path
import librosa

path = Path('/Users/steph/ML_final/subsync/DATA/v1.mp4')

librosa.load(path, sr=16000) # sr is the sampling rate
# %%
import subprocess
import re
import ffmpeg
import librosa

FREQ = 16000
DATA_DIR = 'DATA'
files = os.listdir(DATA_DIR)
p = re.compile('.*.[mp4|avi]')
files = [ f for f in files if p.match(f)]
audio_dirs = []
for f in files:
    name, extension = os.path.splitext(f)
    command = "ffmpeg -1 {0}{1}{2} -ab 160k -ac 2 -ar {3} -vn {0}/{1}_{3}.wav".format(DATA_DIR, name, extension, FREQ)
    # process = (
    #           ffmpeg.input(f)
    #                 .output(name, format='wav')
    #                 .run_async(pipe_stdin=False, cmd='ffmpeg'))
    # out, err = process.communicate()
    # audio_dirs.append(DATA_DIR + name + '_' + str(FREQ) + '.wav')
    audio_dirs.append(os.path.join(DATA_DIR, name + '_' + str(FREQ) + '.wav'))
    subprocess.call(command, shell=True)

    # convert_cmd = [
    #     'ffmpeg',
    #     '-y', # overwrite if exists
    #     '-loglevel', 'error',
    #     '-i', f, # input
    #     '-vn', # no video
    #     '-sn', # no subtitles
    #     '-ac', '1', # convert to mono
    #     name + '.wav'
    # ]
    # 'ffmpeg -i DATA/v1.mp4 -vn -sn v1.wav'
    # subprocess.call(convert_cmd)
    # try:
    # except:
    #     print('error')


    # print('done')

# %%
import time
import librosa


t = time.time()

audio_dir = './DATA/v1.wav'
y, sr = librosa.load(audio_dir, sr=FREQ)
t_audio_load = time.time()-t
    
t = time.time()

# Get MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(HOP_LEN), n_mfcc=N_MFCC)

# %%
HOP_LEN = 512.0
N_MFCC = 13

mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(HOP_LEN), n_mfcc=N_MFCC)

# %%
import numpy as np
import pysrt
from pathlib import Path

if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name

from .audio_converter import *

FREQ = 16000 # Audio frequency
N_MFCC = 13
HOP_LEN = 512.0 # Number of items per sample
ITEM_TIME = (1.0/FREQ)*HOP_LEN

LEN_SAMPLE = 0.5 # Num seconds for input samples 
STEP_SAMPLE = 0.25 # Space between each sample
LEN_MFCC = LEN_SAMPLE/(HOP_LEN/FREQ) # Num of samples to get LEN_SAMPLE
STEP_MFCC = STEP_SAMPLE/(HOP_LEN/FREQ) # Num of samples to get STEP_SAMPLE

samples = mfcc

train_file = 'v1'
subs = pysrt.open(os.path.join(DATA_DIR, train_file)+'.srt', encoding='iso-8859-1')


samples = []
for i in np.arange(0, mfcc.shape[1], STEP_MFCC):
    samples.append(mfcc[:,int(i):int(i)+int(LEN_MFCC)])
    
# Remove last element. Probably not complete
samples = samples[:int((mfcc.shape[1]-LEN_MFCC)/STEP_MFCC)+1]

train_data = np.stack(samples)

labels = np.zeros(len(train_data))

for sub in subs:
        for i in np.arange(timeToPos(sub.start, STEP_MFCC, FREQ, HOP_LEN), timeToPos(sub.end, STEP_MFCC, FREQ, HOP_LEN)+1):
            if i < len(labels):
                labels[i] = 1
            
# %%
import librosa
from pathlib import Path

path = Path('DATA/v1.wav')
librosa.load(path)
# %%
import subprocess

DATA_DIR = '/Users/steph/ML_final/subsync/DATA'
name = 'v1'
command = [
            'ffmpeg',
            '-y', # overwrite if exists
            '-loglevel', 'error',
            '-i', 'DATA/v1.mp4', # inputa
            '-vn', # no video
            '-sn', # no subtitles
            '-ac', '1', # convert to mono
            os.path.join(DATA_DIR, name + '.wav')
        ]
# command = 'ffmpeg -i v1.mp4 -vn -sn v1.wav'
subprocess.check_output(command)
# %%

subprocess.check_output(['echo', 'Hello world!'])
# %%
import pickle
from pathlib import Path
path = 'datasets/dataset_CUT_4000_128.0_2.34375_1.5625.pickle'
# path = Path('datasets/dataset_CUT_4000_128.0_2.34375_1.5625.pickle')
with open(path, 'rb') as f:
    X, Y = pickle.load(f) 
# X, Y = 
# %%
