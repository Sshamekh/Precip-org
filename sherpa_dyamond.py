import numpy as np
import pandas as pd

# machine learning
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LearningRateScheduler,Callback,EarlyStopping
from tensorflow.keras import Model
import xarray as xr
import cartopy.crs as ccrs
from keras.layers.normalization import layer_normalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from netCDF4 import Dataset
# from con1D_model import *
from tensorflow.keras.layers import Dense, Flatten, Layer,Conv1D, Input,Lambda,LeakyReLU,Reshape,Dropout
from tensorflow.keras.initializers import Constant

from netCDF4 import Dataset
import flask
from read_data import get_data

from tqdm import tqdm
# from sherpa import Continuous
import sherpa
import scipy.io

inputs = np.stack(get_data('SAM',200,['qvi','ts','huss','tas'],scaling_method = 'minstd'))
outputs = np.stack(get_data('SAM',200,['pracc'],scaling_method = 'none'))

outputs = np.moveaxis(outputs,0,3)
inputs  = np.moveaxis(inputs,0,3)

mask =  np.random.rand(inputs.shape[0]) < 0.70

input_train,input_test = inputs[mask,:,:,:], inputs[~mask,:,:,:]
nt,nlat,nlon,nfield = input_train.shape
x_train = np.reshape(input_train,(nt*nlat*nlon,nfield))
nt,nlat,nlon,nfield = input_test.shape
x_test = np.reshape(input_test,(nt*nlat*nlon,nfield))




output_train,output_test = outputs[mask,:,:,:], outputs[~mask,:,:,:]
nt,nlat,nlon,nfield = output_train.shape
y_train = np.reshape(output_train,(nt*nlat*nlon,nfield))
nt_test,nlat,nlon,nfield = output_test.shape
y_test = np.reshape(output_test,(nt_test*nlat*nlon,nfield))


# parameters = [sherpa.Continuous(name='lr', range=[0.00001, 0.01], scale='log'), 
#               sherpa.Continuous(name='dropout', range=[0., 0.2]),
#               sherpa.Ordinal(name='batch_size', range=[128,256,]),
#               sherpa.Discrete(name='num_units1', range=[256,200,180,128]),
#               sherpa.Discrete(name='num_units2', range=[200,180,128,64]),
#               sherpa.Discrete(name='num_units3', range=[128,96,64,32])]


parameters = [sherpa.Continuous(name='lr', range=[0.001, 0.01], scale='log'), 
#               sherpa.Continuous(name='dropout', range=[0., 0.2]),
              sherpa.Ordinal(name='batch_size', range=[128,256]),
              sherpa.Discrete(name='num_units1', range=[128,256]),
              sherpa.Discrete(name='num_units2', range=[128,200]),
              sherpa.Discrete(name='num_units3', range=[64,96])]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=100)
study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=True)

# from tensorflow import keras
i=0
act = tf.nn.leaky_relu
for trial in tqdm (study):
        bs = trial.parameters['batch_size']
        model = Sequential()
        model.add(Dense(units=trial.parameters['num_units1'], input_shape=(4,), activation=act))
        model.add(Dense(units=trial.parameters['num_units1'], activation=act))
#         model.add(Dropout(trial.parameters['dropout']))
        model.add(Dense(units=trial.parameters['num_units2'], activation= act))
#         model.add(Dropout(trial.parameters['dropout']))
        model.add(Dense(units=trial.parameters['num_units3'], activation= act))
        model.add(Dense(y_train.shape[1])) 

        adm = keras.optimizers.Adam(lr=trial.parameters['lr']) #Adam
        model.compile(loss='mean_squared_error', optimizer=adm, metrics=['mse'])
        history = model.fit(x_train, y_train, epochs=10, batch_size=bs, verbose = 0,
                            callbacks=[study.keras_callback(trial, objective_name='val_loss')])
        i+=1

        study.get_best_result()
        study.finalize(trial)
        study.save('/glade/scratch/sshamekh/sherpa_outputsuw' )
study.get_best_result()
