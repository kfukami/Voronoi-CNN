# Voronoi-CNN-ch2Dxysec.py
# 2021 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Voronoi CNN for channel flow data.
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
import pickle
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


# Data can be downloaded from https://drive.google.com/drive/folders/1xIY_jIu-hNcRY-TTf4oYX1Xg4_fx8ZvD?usp=sharing


x_num=256
y_num=96
#--- Prepare coordinate ---#
xcor = pd.read_csv('./record_x.csv',header=None,delim_whitespace=False)
ycor = pd.read_csv("./record_y.csv", header=None, delim_whitespace=True)
xcor = xcor.values
ycor = ycor.values
xc=xcor[0:128,0]
yc=ycor[0:48,0]


datasetSerial = np.arange(20,200020,20)## 10000 snapshots

print(len(datasetSerial))

gridSetting = (256,96)
x_num = 128; y_num = 48;

omg_flc_all = []
filename="./ch_2Dxysec.pickle" 
with open(filename, 'rb') as f:
    obj = pickle.load(f)
    omg_flc_all=obj
from scipy.interpolate import griddata
dim_1 = 128
dim_2 = 48

sen_num_kind = 3
sen_num_var = 5
sen_num_kind_list = [50, 100, 200]
sen_num_var_list = [300, 100, 200, 1, 2]


width = dim_1
height = dim_2
x_ref, y_ref =np.meshgrid(yc,xc)
X_ki = np.zeros((10000*sen_num_kind*sen_num_var,dim_1,dim_2,2))
y_ki = np.zeros((10000*sen_num_kind*sen_num_var,dim_1,dim_2,1))
        
for ki in tqdm(range(sen_num_kind)):
    sen_num = sen_num_kind_list[ki]
    
    X_va = np.zeros((10000*sen_num_var,dim_1,dim_2,2))
    y_va = np.zeros((10000*sen_num_var,dim_1,dim_2,1))
    for va in range(sen_num_var):
        
        X_t = np.zeros((10000,dim_1,dim_2,2))
        y_t = np.zeros((10000,dim_1,dim_2,1))
        
        for t in tqdm(range(10000)):
            y_t[t,:,:,0] = omg_flc_all[t,:,:,0]
            np.random.seed(sen_num_var_list[va])
            sparse_locations_lat = np.random.randint((dim_1),size=(sen_num)) # 15 sensors
            sparse_locations_lon = np.random.randint((dim_2),size=(sen_num)) # 15 sensors

            sparse_locations = np.zeros((sen_num,2))
            sparse_locations[:,0] = sparse_locations_lat
            sparse_locations[:,1] = sparse_locations_lon
            

            sparse_data = np.zeros((sen_num))
            for s in range(sen_num):
                sparse_data[s] = (omg_flc_all[t,:,:,0][int(sparse_locations[s,0]),int(sparse_locations[s,1])])
    
            
            sparse_locations_ex = np.zeros(sparse_locations.shape)
            for i in range(sen_num):
                sparse_locations_ex[i,0] = xc[int(sparse_locations[i,0])]
                sparse_locations_ex[i,1] = yc[int(sparse_locations[i,1])]
            grid_z0 = griddata(sparse_locations_ex, sparse_data, (y_ref, x_ref), method='nearest')
            X_t[t,:,:,0] = grid_z0
            mask_img = np.zeros(grid_z0.shape)
            for i in range(sen_num):
                mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
            X_t[t,:,:,1] = mask_img
        
        X_va[10000*va:10000*(va+1),:,:,:] = X_t
        y_va[10000*va:10000*(va+1),:,:,:] = y_t
    X_ki[(10000*sen_num_var)*ki:(10000*sen_num_var)*(ki+1),:,:,:] = X_va
    y_ki[(10000*sen_num_var)*ki:(10000*sen_num_var)*(ki+1),:,:,:] = y_va  
    
    
act = 'relu'
input_img = Input(shape=(dim_1,dim_2,2))
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
X_train, X_test, y_train, y_test = train_test_split(X_ki, y_ki, test_size=0.3, random_state=None)
model_cb=ModelCheckpoint('./Model_2Dxysec.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,y_train,nb_epoch=5000,batch_size=256,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_2Dxysec.csv',index=False)





