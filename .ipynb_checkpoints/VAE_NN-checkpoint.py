from utils_vae import read_field
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow
from tensorflow.keras.losses import mse
from tensorflow.keras import layers
from VAES import dense_gen,encoder_dense

path = "/glade/scratch/sshamekh/dyamond/SAM_highres/"


# %%time
import sys, importlib
importlib.reload(sys.modules['utils_vae'])
from utils_vae import read_field

high_res = 4 # km 
large_scale = 100 #km

varlist_inputs = ['qvi','ts', 'tas','huss']
varlist_outputs = ['clt','pracc']

inputs = read_field(path,varlist_inputs,high_res,large_scale,method = 'minmax',data_reshape = False) 
outputs = read_field(path,varlist_outputs,high_res,large_scale,method = 'None',data_reshape = False)

ng = 25
mask =  np.random.randint(low = 0,high=100,size=inputs.shape[0])<80

train_data = inputs[mask,:,:,:,:,:]
nt = train_data.shape[0]
train_data = train_data.reshape(nt*train_data.shape[1]*train_data.shape[2],ng,ng,4)
print (train_data.shape)

train_out = outputs[mask,:,:,:,:,:]
train_out = train_out.reshape(nt*train_out.shape[1]*train_out.shape[2],ng,ng,2)
print (train_out.shape)


test_data = inputs[~mask,:,:,:,:,:]
nt = test_data.shape[0]
test_data = test_data.reshape(nt*test_data.shape[1]*test_data.shape[2],ng,ng,4)
test_data.shape
test_out = outputs[~mask,:,:,:,:,:]
test_out = test_out.reshape(nt*test_out.shape[1]*test_out.shape[2],ng,ng,2)
print (test_data.shape,test_out.shape)

configs_encoder = {"activation": "relu",
        "latent_dim": 4,
        "conv_1": {
            "filter_num": 32,
            "kernel_size": [4, 4],
            "stride": 2},
        "conv_2": {
            "filter_num": 128,
            "kernel_size": [4, 4],
            "stride": 2},
        "conv_3": {
            "filter_num": 32,
            "kernel_size": [4, 4],
            "stride": 2
        },
        "dense_1": {
            "unit_num":512
                  },
        "dense_2": {
            "unit_num":256
                  },                   
        "dense_3": {
            "unit_num":128
                  },
        "dense_4": {
            "unit_num":64
                  },
        "dense_5": {
            "activation":'sigmoid'
                  },        
                  }
configs_decoder = {
        "latent_dim": 32,
        "activation": "relu",
    
    "dense1":{
        "unit_num":64
        
    },
    "dense2":{
        "unit_num":256,
        'newshape':(8,8,4)
    },
        "conv_t_1": {
            "filter_num": 64,
            "kernel_size": [4, 4],
            "stride": 2
        },
        "conv_t_2": {
            "filter_num": 128,
            "kernel_size": [4, 4],
            "stride": 2
        },
        "conv_t_3": {
            "filter_num": 32,
            "kernel_size": [4, 4],
            "stride": 2,
        },
    
        "conv_t_4": {
            "filter_num": 1,
            "kernel_size": [4, 4],
            "stride": 1,
            "activation": "elu"
        }
    }

config_dense  = {
    'activation' : 'relu',
     'layer_1':{
         "unit_num":265,
     },
     'layer_2':{
         "unit_num":265,
     },
     'layer_3':{
         "unit_num":128,
     },
     'layer_4':{
         "unit_num":64,
     },
         'layer_5':{
         "unit_num":1,
         "activation": None
     }
}


x_train_lg = np.mean(train_data,(1,2))
x_test_lg = np.mean(test_data,(1,2))
y_train_lg = np.mean(train_out,(1,2))
y_test_lg = np.mean(test_out,(1,2))

inds_train = np.where(y_train_lg[:,1]>0.05)[0]
inds_test = np.where(y_test_lg[:,1]>0.05)[0]

x_train_hr = train_data[inds_train,:]
x_test_hr = test_data[inds_test,:]
x_train_lg = x_train_lg[inds_train,:]
x_test_lg = x_test_lg[inds_test,:]
y_train_lg = y_train_lg[inds_train,:]
y_test_lg = y_test_lg[inds_test,:] 


del train_data, test_data, train_out,test_out

encoder_result = encoder_dense((25, 25,4),configs_encoder)
# vae_decoder = decoder_dense((32,32,1),configs_decoder)
z = encoder_result.vae_encoder(encoder_result.inputs)

inshape = (4,)
inputdense = layers.Input(shape=inshape)
zz = layers.Concatenate(axis = -1)([z,inputdense])

dense_model = dense_gen((8,),config_dense)
precip = dense_model(zz)
vae = Model(inputs=[encoder_result.inputs,inputdense], outputs=[precip])

optimizer = tensorflow.keras.optimizers.Adam(lr=0.0001)
earlyStopping=tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
vae.compile(loss = mse, optimizer = optimizer)

hist = vae.fit(
        x=[x_train_hr,x_train_lg], 
        y=y_train_lg[:,1:], 
        epochs=15,
        batch_size=128,
        validation_split = 0.2,
        callbacks= [earlyStopping],
        shuffle = True
    )

predict = vae.predict([x_test_hr,x_test_lg])


import utils

def plot_pw_var():
    avg_pw,avg_pred,std_pw,std_pred = utils.bin_threshold(x_test_lg[:,0],predict[:,0],50)
    avg_pw_lg,avg_pred_lg,std_pw_lg,std_pred_lg = utils.bin_threshold(x_test_lg[:,0],predict_lg[:,0],50)
    avg_pw,avg_pred_clt,std_pw_clt,std_pred_clt = utils.bin_threshold(x_test_lg[:,0],y_test_lg[:,1],50)

    fig = plt.figure(figsize=[10,8])

    plt.fill_between(np.array(avg_pw),np.array(avg_pred_lg)-np.array(std_pred_lg),
                     np.array(avg_pred_lg)+np.array(std_pred_lg),alpha = 0.5)
    plt.plot(np.array(avg_pw),avg_pred_lg,color = 'navy',linestyle = ':',label = 'pred_lg',linewidth=3)


    plt.fill_between(np.array(avg_pw),np.array(avg_pred_clt)-np.array(std_pred_clt),
                     np.array(avg_pred_clt)+np.array(std_pred_clt),alpha = 0.5,color = 'orange')
    plt.plot(np.array(avg_pw),avg_pred_clt,color = 'orange',linestyle = '-',label = 'True'
            ,linewidth=3)

    plt.fill_between(np.array(avg_pw),(np.array(avg_pred)-np.array(std_pred)),
                     (np.array(avg_pred)+np.array(std_pred)),alpha = 0.5,color='green')
    plt.plot(np.array(avg_pw),np.array(avg_pred),color = 'green',linestyle = '-.',label = 'Prediction hr'
            ,linewidth=3)
    plt.legend(loc = 2,fontsize =16)
    plt.ylabel('Precip error',fontsize = 20)
    plt.xlabel('PW',fontsize = 20)
    plt.xlim()
    # plt.ylim(-2,4)
    # plt.xlim(0.,4)
    # 
    plt.tick_params(axis='both', which='major', labelsize=20)
    fig.savefig('pw_var_vae_nn.jpeg',bbox_inches = 'tight',dpi = 100)
    

vae.save('vae_nn')
dense_model.save('dense_vae')
encoder_result.save('encoder_vae')
plot_pw_var()