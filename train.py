# %%
import time
import sys
from pathlib import Path
from numpy import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, BatchNormalization

if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .audio_converter import *

def model_lstm(input_shape):
    model = tf.keras.Sequential()
    model.add(LSTM(16, activation='relu', input_shape=(input_shape)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_lstm():
    t = time.time()
    
#    freq = 16000.0
#    hop_len = 128.0
#    len_sample = 0.25    # Length in seconds for the input samples
#    step_sample = 0.05    # Space between the beginingof each sample
    step_sample = 0.05    # Space between the beginingof each sample
    
    train_files = ['v1', 'v2', 'v3', 'v4']
    
#    for len_sample in [0.5, 0.25, 0.125, 0.075]:
    for len_sample in [0.075]:
#        for f in [1000, 2000, 4000, 8000, 16000]:
        for f in [4000, 8000, 16000]:
            for hop_len in [128.0, 256.0, 512.0, 1024.0, 2048.0]:
                
                print ('FREQ:',  f, hop_len, len_sample)
                
                t = time.time()
                
                len_mfcc = get_len_mfcc(len_sample, hop_len, f)     #  Num of samples to get LEN_SAMPLE
                step_mfcc = get_step_mfcc(step_sample, hop_len, f)     #  Num of samples to get STEP_SAMPLE

                X, Y = generateDatasets(train_files, True, len_mfcc, step_mfcc, hop_len=hop_len, freq=f)

                # rand = np.random.permutation(np.arange(len(Y)))
                rand = random.randint(len(Y))
                X = X[rand]
                Y = Y[rand]
                
                X = np.array([ np.rot90(val) for val in X ])
                X = X - np.mean(X, axis=0)
            #    X = X[:,1:,:]
            
                print (X.shape, len(Y[Y==0]), len(Y[Y==1]), float(len(Y[Y==0]))/len(Y[Y==1]))
                
                if X.shape[1] == 0:
                    print ("NEXT\n")
                    continue
                
            
                input_shape = (X.shape[1], X.shape[2])   
                model = model_lstm(input_shape)
                os.makedirs('models/lstm', exist_ok=True)
                earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=0, mode='min', patience=5)
                filename = 'models/lstm/model_cnn_lstm_' + str(f) + '_' + str(len_mfcc) + '_' + str(step_mfcc) + '_' + str(hop_len) + '.hdf5'
                checkpoint = ModelCheckpoint(filepath=filename, 
                             monitor='val_loss', verbose=0, save_best_only=True)
                callbacks_list = [earlyStopping, checkpoint]
                model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        #        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
            #    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
                hist = model.fit(X, Y, epochs=20, batch_size=32, shuffle=True, validation_split=0.25, verbose=0, callbacks=callbacks_list)
                
                print ('val_loss:', min(hist.history['val_loss']))
                print ('val_acc:', max(hist.history['val_acc']))
                
                print ("Total training time:", (time.time()-t)/60)
                print ("-----------------------------")
                print ("-----------------------------")
                print ("-----------------------------")
                print ("-----------------------------\n\n\n")

train_lstm()
 # %%
