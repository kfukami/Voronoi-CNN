# Voronoi-CNN-NOAA.py
# 2021 Kai Fukami (UCLA, kfukami1@g.ucla.edu)

## Voronoi CNN for NOAA SST data.
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


import h5py
import numpy as np




f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])

sst1 = np.nan_to_num(sst)

sen_num_kind = 5
sen_num_var = 5
sen_num_kind_list = [10, 20, 30, 50, 100]
sen_num_var_list = [300, 100, 200, 1, 2]

X_ki = np.zeros((1040*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))
y_ki = np.zeros((1040*sen_num_kind*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
        
sst_reshape = sst[0,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')
x_ref, y_ref = np.meshgrid(lon,lat)
xv1, yv1 =np.meshgrid(lon[0,:],lat[0,:])

for ki in tqdm(range(sen_num_kind)):
    sen_num = sen_num_kind_list[ki]
    
    X_va = np.zeros((1040*sen_num_var,len(lat[0,:]),len(lon[0,:]),2))
    y_va = np.zeros((1040*sen_num_var,len(lat[0,:]),len(lon[0,:]),1))
    for va in range(sen_num_var):
        
        X_t = np.zeros((1040,len(lat[0,:]),len(lon[0,:]),2))
        y_t = np.zeros((1040,len(lat[0,:]),len(lon[0,:]),1))
        
        for t in tqdm(range(1040)):
            y_t[t,:,:,0] = np.nan_to_num(sst[t,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F'))
            np.random.seed(sen_num_var_list[va])
            sparse_locations_lat = np.random.randint(len(lat[0,:]),size=(sen_num)) # 15 sensors
            sparse_locations_lon = np.random.randint(len(lon[0,:]),size=(sen_num)) # 15 sensors

            sparse_locations = np.zeros((sen_num,2))
            sparse_locations[:,0] = sparse_locations_lat
            sparse_locations[:,1] = sparse_locations_lon

            for s in range(sen_num):
                a = sparse_locations[s,0]
                b = sparse_locations[s,1]
                while np.isnan(sst_reshape[int(a),int(b)]) == True:
                    a = np.random.randint(len(lat[0,:]),size=(1))
                    b = np.random.randint(len(lon[0,:]),size=(1))
                    sparse_locations[s,0] = a
                    sparse_locations[s,1] = b

            sparse_data = np.zeros((sen_num))
            for s in range(sen_num):
                sparse_data[s] = (y_t[t,:,:,0][int(sparse_locations[s,0]),int(sparse_locations[s,1])])
    
            sparse_locations_ex = np.zeros(sparse_locations.shape)
            for i in range(sen_num):
                sparse_locations_ex[i,0] = lat[0,:][int(sparse_locations[i,0])]
                sparse_locations_ex[i,1] = lon[0,:][int(sparse_locations[i,1])]
            grid_z0 = griddata(sparse_locations_ex, sparse_data, (yv1, xv1), method='nearest')
            for j in range(len(lon[0,:])):
                for i in range(len(lat[0,:])):
                    if np.isnan(sst_reshape[i,j]) == True:
                        grid_z0[i,j] = 0
            X_t[t,:,:,0] = grid_z0
            mask_img = np.zeros(grid_z0.shape)
            for i in range(sen_num):
                mask_img[int(sparse_locations[i,0]),int(sparse_locations[i,1])] = 1
            X_t[t,:,:,1] = mask_img
        
        X_va[1040*va:1040*(va+1),:,:,:] = X_t
        y_va[1040*va:1040*(va+1),:,:,:] = y_t
    X_ki[(1040*sen_num_var)*ki:(1040*sen_num_var)*(ki+1),:,:,:] = X_va
    y_ki[(1040*sen_num_var)*ki:(1040*sen_num_var)*(ki+1),:,:,:] = y_va
    

input_img = Input(shape=(len(lat[0,:]),len(lon[0,:]),2))
x = Conv2D(48, (7,7),activation='relu', padding='same')(input_img)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x = Conv2D(48, (7,7),activation='relu', padding='same')(x)
x_final = Conv2D(1, (7,7), padding='same')(x)
model = Model(input_img, x_final)
model.compile(optimizer='adam', loss='mse')


from keras.callbacks import ModelCheckpoint,EarlyStopping
X_train, X_test, y_train, y_test = train_test_split(X_ki, y_ki, test_size=0.3, random_state=None)
model_cb=ModelCheckpoint('./Model_NOAA.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=100,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,y_train,nb_epoch=5000,batch_size=32,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./Model_NOAA.csv',index=False)





