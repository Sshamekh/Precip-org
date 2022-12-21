## data process and plotting import
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib import cm
from tqdm import tqdm
from netCDF4 import Dataset

## costum import 
from con1D_model import get_data

## ml imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Flatten, Dropout,Conv1D,Input,Lambda,LeakyReLU,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation


def schedule(epoch):
    
    
    if epoch < 20:
        return 0.0004
    elif epoch < 30:
        return 0.0002
    
    elif epoch < 50:
        return 0.0001
    
    elif epoch < 100:
        return 0.00007

def get_traintest(case):
    X, Y= get_data(case)


    mask =  np.random.rand(X.shape[0]) < 0.80

    y_train, y_test = Y[mask,:], Y[~mask,:]
    x_train,  x_test  = X[mask,:], X[~mask,:]
    return y_train, y_test,x_train,x_test

class Loss_MSE_kernel( keras.layers.Layer ):
    def __init__(self,name = None):
        super(Loss_MSE_kernel,self).__init__(name= name)
        self.loss_mse = tf.keras.losses.MeanSquaredError()
    
    def call(self,inputs):
        n_dims = 1
        ytrue,ypred = inputs['true'],inputs['pred']
        loss = self.loss_mse(ytrue,ypred)
        self.add_loss(loss)
        self.add_metric(loss,name = self.name)
        
        k_weight = self.get_layer('local').kernel
        constraint = K.sum(K.abs(K.sum(k_weight,axis = -1)))
        
        return ypred
    
    
def costum_loss(kernelt,lamda=100):
    @tf.function
    def losstot(y_true,y_predict):
            print (kernelt.shape)
            loss =  K.mean(K.square(y_true-y_predict)) 
            loss += lamda *K.sum( K.abs(K.sum(kernelt,axis =-1)) )

            return loss
    return losstot



def kernelNNconv(x_train,xcoef_train,y_train,model_config):

    scalarshape = x_train[0,:].shape
    inputshape = xcoef_train[0,:].shape
    print (scalarshape)
    ## scalarshape is (None, nchunck,1)
    scalar_input = Input(shape=(scalarshape),name='indt')

    ## nfilter is the number of filters. should be between 2 to 5
    ## kernel size should match nchunk

    convT = Conv1D(filters=model_config['nfilter'], 
                   kernel_size=model_config['kernel_size'], 
                   kernel_initializer= model_config['initializer']
                   # padding = 'causal',
                   name = 'conv',)(scalar_input)
    print (convT.shape)
    ## output of conT has the shape (None,nfilter,1)

    ## Coefficients 
    coef_input = Input(shape=(inputshape),name = 'krnl_in1')
    xc = Flatten()(coef_input)
    xc = Dense(units=model_config['units1'],activation = model_config['activation'])(xc)
    xc = Dense(units=model_config['units2'],activation = model_config['activation'])(xc)
    xc = Dense(units=model_config['units3'],activation = model_config['activation'])(xc)
    coeffs = Dense(units= model_config['nfilter'], activation=None, name = 'coeff')(xc)

    ## Activate one of the following line if you want the coeffs to be positive. 
    ## This is the case for eddy diffusivity

#     xc = Lambda(lambda x: K.exp(x))(xc) 
#     xc = LeakyReLU(alpha=0.03)(xc)

    ypredict =keras.layers.Dot(name = 'dot',axes = -1)([coeffs,convT])
    print (ypredict.shape)

    model = Model(inputs = [scalar_input,coef_input],outputs = [ypredict])
    opt = keras.optimizers.Adam(learning_rate=0.001)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
    callback_lr=LearningRateScheduler(schedule,verbose=1)
    earlyStopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto')

    weights = model.get_layer('conv').kernel
    model.compile(optimizer = opt,loss = costum_loss(weights))

    hist = model.fit([x_train,xcoef_train],[y_train], epochs = epochs,  validation_split = val_split,
                         batch_size =  batch_size,verbose = 1, shuffle=True)
    return hist, model
