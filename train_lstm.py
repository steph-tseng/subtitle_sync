# %%
import time
import sys
from pathlib import Path
import pickle
from numpy import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Dropout, BatchNormalization

if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
# from .train import *
from .audio_converter import *
import warnings
warnings.filterwarnings('ignore')

path = 'datasets/dataset_CUT_4000_128.0_2.34375_1.5625.pickle'
# path = 'datasets/dataset_CUT_16000_512.0_15.625_7.8125.pickle'

with open(path, 'rb') as f:
    X, Y = pickle.load(f) 

# print(X.shape)
# print(Y.shape)
X = np.array([np.rot90(val) for val in X])
Y = np.reshape(Y, (-1,1))
X = X -np.mean(X, axis =0)

input_shape = (X.shape[1], X.shape[2])
# print(input_shape)

def model_lstm(input_shape):
    model = tf.keras.Sequential()
    model.add(LSTM(8, activation='relu', input_shape=(input_shape)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    return model

t = time.time()
model = model_lstm((input_shape))
os.makedirs('models/lstm', exist_ok=True)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=0, mode='min', patience=5)
filename = 'models/lstm/model_cnn_lstm_' + str(f) + '_' + str(LEN_MFCC) + '_' + str(STEP_MFCC) + '_' + str(HOP_LEN) + '.h5'
checkpoint = ModelCheckpoint(filepath=filename, 
              monitor='val_loss', verbose=0, save_best_only=True)
callbacks_list = [earlyStopping, checkpoint]
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
#        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
#    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
hist = model.fit(X, Y, epochs=2000, shuffle=True, validation_split=0.25, verbose=1, callbacks=callbacks_list)

print ('val_loss:', min(hist.history['val_loss']))
# print ('val_acc:', max(hist.history['val_acc']))

print ("Total training time:", (time.time()-t)/60)
print ("-----------------------------")
print ("-----------------------------")
print ("-----------------------------")
print ("-----------------------------\n\n\n")


# %%
model.save(f'models/lstm/model_{str(f)}_{str(LEN_MFCC)}_{str(STEP_MFCC)}_{str(HOP_LEN)}.h5')


# %%
