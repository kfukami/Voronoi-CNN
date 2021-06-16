# Voronoi-CNN-cy.py
# 2021 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Voronoi CNN for cylinder wake data.
## Authors:
# Kai Fukami (UCLA), Romit Maulik (Argonne National Lab.), Nesar Ramachandra (Argonne National Lab.), Koji Fukagata (Keio University), Kunihiko Taira (UCLA)

## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
# Ref: K. Fukami, R. Maulik, N. Ramachandra, K. Fukagata, and K. Taira,
#     "Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning,"
#     in Review, 2021
#
# The code is written for educational clarity and not for speed.
# -- version 1: Mar 13, 2021

from keras.layers import Input, Add, Dense, Conv2D, merge, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Reshape, LSTM
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from tqdm import tqdm as tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial import Voronoi
import math
from scipy.interpolate import griddata


import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True, 
        visible_device_list="0" 
    )
)
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


datasetSerial = np.arange(90000,100000,2) #5000snapshots

X = np.zeros((len(datasetSerial)*2,112,192,2))
y_1 = np.zeros((len(datasetSerial)*2,112,192,1))


# Data can be downloaded from https://drive.google.com/drive/folders/1K7upSyHAIVtsyNAqe6P8TY1nS5WpxJ2c?usp=sharing

df = pd.read_csv('./cylinder_xx.csv',header=None,delim_whitespace=False)
dataset = df.values
x = dataset[:,:]
x_ref = x[7:119,0:192]
print(x.shape)
df = pd.read_csv('./cylinder_yy.csv',header=None,delim_whitespace=False)
dataset = df.values
y = dataset[:,:]
y_ref = y[7:119,0:192]


omg_box = []
filename="./Cy_Taira.pickle" 
with open(filename, 'rb') as f:
    obj = pickle.load(f)
    omg_box=obj
print(omg_box.shape)
    
    

for t in tqdm(range(len(datasetSerial))):
    omg = omg_box[t,:,:,0]
    y_1[t,:,:,0] = omg
    
    sparse_locations1 = (np.array([[76,71], [175,69],  [138,49],                   
                [41, 56], [141,61] ,[30,41],  
                [177,40],[80,55]]))
    sparse_locations = np.zeros(sparse_locations1.shape)
    sparse_locations[:,0] = sparse_locations1[:,1]
    sparse_locations[:,1] = sparse_locations1[:,0]

    sen_num = 8
    width = 112
    height = 192

    sparse_data = np.zeros((sen_num)) 
    sparse_data[0] = (omg[71,76])
    sparse_data[1] = (omg[69,175])
    sparse_data[2] = (omg[49,138])
    sparse_data[3] = (omg[56,41])
    sparse_data[4] = (omg[61,141])
    sparse_data[5] = (omg[41,30])
    sparse_data[6] = (omg[40,177])
    sparse_data[7] = (omg[55,80])


    sparse_locations_ex = np.zeros(sparse_locations.shape)
    for i in range(sen_num):
        sparse_locations_ex[i,0] = y_ref[:,0][int(sparse_locations[i,0])]
        sparse_locations_ex[i,1] = x_ref[0,:][int(sparse_locations[i,1])]
    grid_z0 = griddata(sparse_locations_ex, sparse_data, (y_ref, x_ref), method='nearest')
    X[t,:,:,0] = grid_z0
    
    mask_img = np.zeros(grid_z0.shape)
    for i in range(sen_num):
        mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
    X[t,:,:,1] = mask_img

for t in tqdm(range(len(datasetSerial))):
    omg = omg_box[t,:,:,0]
    y_1[len(datasetSerial)+t,:,:,0] = omg
    
    sparse_locations1 = (np.array([[76,71], [175,69],  [138,49],                   
                    [41, 56], [141,61] ,[30,41],  
                    [177,40],[80,55], [60,41],[70,60],
                    [100,60],[120,51],[160,80],[165,50],[180,60],[30,70]      
                              ]))
    sparse_locations = np.zeros(sparse_locations1.shape)
    sparse_locations[:,0] = sparse_locations1[:,1]
    sparse_locations[:,1] = sparse_locations1[:,0]

    sen_num = 16
    width = 112
    height = 192

    sparse_data = np.zeros((sen_num)) 
    sparse_data[0] = (omg[71,76])
    sparse_data[1] = (omg[69,175])
    sparse_data[2] = (omg[49,138])
    sparse_data[3] = (omg[56,41])
    sparse_data[4] = (omg[61,141])
    sparse_data[5] = (omg[41,30])
    sparse_data[6] = (omg[40,177])
    sparse_data[7] = (omg[55,80])
    sparse_data[8] = (omg[41,60])
    sparse_data[9] = (omg[60,70])
    sparse_data[10] = (omg[60,100])
    sparse_data[11] = (omg[51,120])
    sparse_data[12] = (omg[80,160])
    sparse_data[13] = (omg[50,165])
    sparse_data[14] = (omg[60,180])
    sparse_data[15] = (omg[70,30])


    sparse_locations_ex = np.zeros(sparse_locations.shape)
    for i in range(sen_num):
        sparse_locations_ex[i,0] = y_ref[:,0][int(sparse_locations[i,0])]
        sparse_locations_ex[i,1] = x_ref[0,:][int(sparse_locations[i,1])]
    grid_z0 = griddata(sparse_locations_ex, sparse_data, (y_ref, x_ref), method='nearest')
    X[len(datasetSerial)+t,:,:,0] = grid_z0
    
    mask_img = np.zeros(grid_z0.shape)
    for i in range(sen_num):
        mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
    X[len(datasetSerial)+t,:,:,1] = mask_img


input_img = Input(shape=(112,192,2))
x = Conv2D(48, (7,7),activation='relu', padding='same')(input_img)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x_final = Conv2D(1, (3,3), padding='same')(x)
model = Model(input_img, x_final)
model.compile(optimizer='adam', loss='mse')


from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size=0.3, random_state=None)
model_cb=ModelCheckpoint('./Model_cy.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,y_train,nb_epoch=5000,batch_size=128,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_cy.csv',index=False)





