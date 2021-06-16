# All credit to Alberto Sabeter, I've link his github and blog in my README, it's just updated a bit so that it works with python 3.9
# %%
import numpy as np
import subprocess
import librosa
import pysrt
import pickle
import os
import re
import time
import tensorflow as tf

DATA_DIR = 'DATA'
STORE_DIR = 'datasets'

#Default variables
FREQ = 16000 # Audio frequency
N_MFCC = 13
HOP_LEN = 512.0 # Number of items per sample
ITEM_TIME = (1.0/FREQ)*HOP_LEN

LEN_SAMPLE = 0.5 # Num seconds for input samples 
STEP_SAMPLE = 0.25 # Space between each sample
LEN_MFCC = LEN_SAMPLE/(HOP_LEN/FREQ) # Num of samples to get LEN_SAMPLE
STEP_MFCC = STEP_SAMPLE/(HOP_LEN/FREQ) # Num of samples to get STEP_SAMPLE

def get_len_mfcc(len_sample, hop_len, freq):
    return len_sample/(hop_len/freq)

def get_step_mfcc(step_sample, hop_len, freq):
    return step_sample/(hop_len/freq)

def getAudio(freq=FREQ, audio_files=None):
    files = os.listdir(DATA_DIR)
    p = re.compile('.*mp4')
    files = [ f for f in files if p.match(f)] # Take only files that have .mp4 extension

    if audio_files:
        files = [ f for f in files if os.path.splitext(f)[0] in audio_files] # Take only first instance of the file returned
    audio_dirs = [] # Returns a list of .wav files that have been converted from .mp4
    for f in files:
        name, extension = os.path.splitext(f) # Split name and extension
        command = [
            'ffmpeg',
            '-n', # ignore if exists
            # '-y', # overwrite if exists
            '-loglevel', 
            'error',
            '-i', os.path.join(DATA_DIR, f), # input
            '-vn', # no video
            '-sn', # no subtitles
            '-ac', '1', # convert to mono
            os.path.join(DATA_DIR, name + '.wav')
        ]
        # command = 'ffmpeg -i v1.mp4 -vn -sn v1.wav'
        audio_dirs.append(os.path.join(DATA_DIR, name + '.wav')) 
        subprocess.check_output(command) # Runs command variable (as defined above) to convert file from .mp4 to .wav using ffmpeg
    
    return audio_dirs

# Convert timestamp to seconds
def timeToSec(t):
    total_sec = float(t.milliseconds)/1000
    total_sec += t.seconds
    total_sec += t.minutes*60
    total_sec += t.hours*60*60
    
    return total_sec
    

# Return timestamp from cell position
def timeToPos(t, step_mfcc, freq, hop_len):
    return int((float(freq*timeToSec(t))/hop_len)/step_mfcc)

# Return seconds from cell position
def secToPos(t, step_mfcc, freq, hop_len):
    return int((float(freq*t)/hop_len)/step_mfcc)

# Return cell position from timestamp
def posToTime(pos, step_mfcc, freq, hop_len):
    return float(pos)*step_mfcc*hop_len/freq


# Generate a train dataset from a video file and its subtitles
def generateSingleDataset(train_file, cut_data, len_mfcc, step_mfcc, hop_len, freq, verbose=False):
    if verbose: print('Loading', train_file, 'data...')
    
    total_time = time.time()
    
    # Process audio
    t = time.time()
    audio_dir = getAudio(freq, [train_file])[0]
    print('audio dir', audio_dir)
    t_audio = time.time()-t
    if verbose: print("- Audio extracted: {0:02d}:{1:02d}").format(int(t_audio/60), int(t_audio % 60))

    # Load subtitles
    subs = pysrt.open(os.path.join(DATA_DIR, train_file+'.srt'), encoding='iso-8859-1')
    t = time.time()
   
    if cut_data:
        # Def audio starting and ending point
        start = timeToSec(subs[0].start)-2
        start = start if start>0 else 0
        subs.shift(seconds=-start)
        
        # Load audio file
        y, sr = librosa.load(audio_dir, sr=freq, offset=start)#, duration=end)
    else:
        # Load audio file
        y, sr = librosa.load(audio_dir, sr=freq)
    # Remove audio from disk
    os.remove(audio_dir)
    
    t_audio_load = time.time()-t
    if verbose: print("- Audio loaded: {0:02d}:{1:02d}").format(int(t_audio_load/60), int(t_audio_load % 60))

    t = time.time()

    # Get MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=int(hop_len), n_mfcc=N_MFCC)
    
    if len_mfcc==1 and step_mfcc ==1:
        samples = []
        for i in np.arange(0, mfcc.shape[1], step_mfcc):
            samples.append(mfcc[:,int(i):int(i)+int(len_mfcc)])
            
        # Remove last element. Probably not complete
        samples = samples[:int((mfcc.shape[1]-len_mfcc)/step_mfcc)+1]
       
        train_data = np.stack(samples)
    else:
        samples = mfcc
        train_data = np.stack(samples)
    t_feat = time.time()-t
    if verbose: print("- Features calculated: {0:02d}:{1:02d}").format(int(t_feat/60), int(t_feat % 60))


    t = time.time()

    # Create array of labels
    labels = np.zeros(shape=(train_data.shape))
    for sub in subs:
        for i in np.arange(timeToPos(sub.start, step_mfcc, freq, hop_len), timeToPos(sub.end, step_mfcc, freq, hop_len)+1):
            if i < len(labels):
                labels[i] = 1
            
    t_labels = time.time()-t
    if verbose: print("- Labels calculated: {0:02d}:{1:02d}").format(int(t_labels/60), int(t_labels % 60))
    total_time = time.time()-total_time
    # print ('Data times. audio: {0:.2f}, audio_load {1:.2f}, t_feat: {2:.2f},  t_labels {3:.2f}, total_time: {4:.2f}, {5}').format(t_audio, t_audio_load, t_feat, t_labels, total_time, str(train_data.shape))
    
    if verbose: print(train_data.shape, labels.shape)
    return train_data, labels
    
    
# Generate a train dataset from an array of video files
def generateDatasets(train_files, cut_data, len_mfcc, step_mfcc, hop_len, freq):
    print('train files', train_files)
    
    X, Y = [], []
    
    for t_f in train_files:

        train_data, labels = generateSingleDataset(t_f, cut_data, len_mfcc, step_mfcc, hop_len, freq)
                
        X.append(train_data)
        Y.append(labels)
    
    # X = tf.ragged.constant(X)
    # X = X.to_tensor()
    # Y = tf.ragged.constant(Y)
    # Y = Y.to_tensor()
    # X = tf.concat(X)
    # Y = tf.concat(Y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    # X = tf.data.Dataset.from_tensor_slices(X)
    # Y = tf.data.Dataset.from_tensor_slices(Y)

    os.makedirs(STORE_DIR, exist_ok=True)
    if cut_data:
        # filename = STORE_DIR + 'dataset_CUT_' + str(freq) + '_' + str(hop_len) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '_' + str(X.shape[0]) + '_' + str(X.shape[1]) + '_' + str(X.shape[2]) + '.pickle'
        # filename = STORE_DIR + 'dataset_CUT_' + str(freq) + '_' + str(hop_len) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '.pickle'
        filename = os.path.join(STORE_DIR, f'dataset_CUT_{freq}_{hop_len}_{len_mfcc}_{step_mfcc}.pickle')
    else:
        # filename = STORE_DIR + 'dataset_' + str(freq) + '_' + str(hop_len) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '_' + str(X.shape[0]) + '_' + str(X.shape[1]) + '_' + str(X.shape[2]) + '.pickle'
        # filename = STORE_DIR + 'dataset_' + str(freq) + '_' + str(hop_len) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '.pickle'
        filename = os.path.join(STORE_DIR, f'dataset_{freq}_{hop_len}_{len_mfcc}_{step_mfcc}.pickle')


    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump([X, Y], f)
        
        
    return X, Y

# Generate a dataset from all available files
def generateAllDatasets(freq=FREQ):
    
    train_files = ['v1', 'v2', 'v3', 'v4']
    cut_data = True
    len_sample = 0.5    # Length in seconds for the input samples
    step_sample = 0.05    # Space between the beginingof each sample
    
    len_mfcc = get_len_mfcc(len_sample, HOP_LEN, freq)     #  Num of samples to get LEN_SAMPLE
    step_mfcc = get_step_mfcc(step_sample, HOP_LEN, freq)     #  Num of samples to get STEP_SAMPLE
    X, Y = generateDatasets(train_files, cut_data, len_mfcc, step_mfcc, freq=freq)

    return X, Y


def generateTestDatset():
    generateDatasets(['v1'], cut_data=False, len_mfcc=15.625, step_mfcc=7.8125, freq=FREQ)

    
def testDatasetTimes():
    
    f = 16000.0
    hop_len = 128.0
    len_sample = 0.25    # Length in seconds for the input samples
    step_sample = 0.1    # Space between the beginingof each sample
                
    
    train_files = ['v1', 'v2', 'v3', 'v4']
    
    for tf in train_files:
        print('\n * ', tf)
        print('_________')
#        for f in [2000, 4000, 8000, 16000, 22050]:
        for len_sample in [0.075,0.125,0.25,0.5]:
            print('Freq:', f)
            len_mfcc = get_len_mfcc(len_sample, hop_len, f)     #  Num of samples to get LEN_SAMPLE
            step_mfcc = get_step_mfcc(step_sample, hop_len, f)     #  Num of samples to get STEP_SAMPLE
        
#            X, Y = generateDatasets(train_files, True, len_mfcc, step_mfcc, hop_len=hop_len, freq=f)
            X, Y = generateSingleDataset(tf, True, len_mfcc, step_mfcc, hop_len=hop_len, freq=f)
            



# %%
