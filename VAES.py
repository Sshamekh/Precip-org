import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import netCDF4

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mse


from tensorflow.keras import regularizers
from tensorflow.keras import initializers


#initializer = tf.keras.initializers.GlorotNormal()
initializer = None

def loss(true, pred):
    return mse(true, pred)

class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, epochs):
        super(AnnealingCallback, self).__init__()
        self.epochs = epochs 
        
    def on_epoch_begin(self, epoch, logs={}):
        new_kl_weight = epoch/self.epochs 
        K.set_value(self.model.kl_weight, new_kl_weight)
        print("Using updated KL Weight:", K.get_value(self.model.kl_weight))

def schedule(epoch):
       
    if epoch < 20:
        return 0.0004
    elif epoch < 30:
        return 0.0002
    
    elif epoch < 50:
        return 0.0001
    
    elif epoch < 100:
        return 0.00007
    
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        """
        Add descriptions 
        """
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var/2) + mean

def kl(z_log_var, z_mean):
    def _kl(true, pred):
        """
        KL divergence loss
        To be used if VAE
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(kl_loss)
    
    return _kl

def encoder_gen(input_shape: tuple, encoder_config: dict):
    """
    Encoder with three convolutional layer
    """


    class EncoderResult():
        pass 

    encoder_result = EncoderResult()
    
    # Construct VAE Encoder layers
    inputs = layers.Input(shape=input_shape)
    zero_padded_inputs = layers.ZeroPadding2D(padding=(1, 0))(inputs)

    print("shape of input after padding", inputs.shape)
    
    x = layers.Conv2D(
        encoder_config["conv_1"]["filter_num"], 
        tuple(encoder_config["conv_1"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_1"]["stride"]
    )(zero_padded_inputs)
    print("shape after first convolutional layer", x.shape)

    x = layers.Conv2D(
        encoder_config["conv_2"]["filter_num"], 
        tuple(encoder_config["conv_2"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_2"]["stride"]
    )(x)

    print("shape after second convolutional layer", x.shape)

    x = layers.Conv2D(
        encoder_config["conv_3"]["filter_num"], 
        tuple(encoder_config["conv_3"]["kernel_size"]), 
        padding='same', 
        activation=encoder_config["activation"], 
        strides=encoder_config["conv_3"]["stride"]
    )(x)


    shape_before_flattening = K.int_shape(x) 
    print("shape before flattening", x.shape)

    x = keras.layers.Flatten()(x)
    x = layers.Dense(encoder_config["dense_2"]["unit_num"],activation=encoder_config["activation"])(x)
#     # Compute latent state 
    xout = layers.Dense(encoder_config["latent_dim"], name='xmean')(x)
    
#     x_log_var = layers.Dense(encoder_config["latent_dim"], name='x_logvar')(x)
#     xout = Sampling()([x_mean, x_log_var])
    print("shape of the output", xout.shape)

    # Instantiate Keras model for VAE encoder 
    vae_encoder = Model(inputs = [inputs], outputs=[xout])

    # Package up everything for the encoder
    encoder_result.inputs = inputs
    encoder_result.x = x
    encoder_result.xout = xout
    encoder_result.vae_encoder = vae_encoder 
    encoder_result.shape_before_flattening = shape_before_flattening

    return encoder_result

def decoder_gen(
    original_input: tuple,
    decoder_config: dict, 
    shape_before_flat: tuple,
    multiple_dim: int):

    """
    Decoder with convolutional layer
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    
    # x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=decoder_config["activation"])(decoder_inputs)
    x = keras.layers.Dense(multiple_dim)(decoder_inputs)

    # Reshape input to 2D
    # x = keras.layers.Reshape(shape_before_flat[1:])(x)
    x = keras.layers.Reshape(shape_before_flat)(x)

    # Start tranpose convolutional layers that upsample the image
    print("shape at beginning of decoder", x.shape)

    x = layers.Conv2DTranspose(
        decoder_config["conv_t_1"]["filter_num"], 
        tuple(decoder_config["conv_t_1"]["kernel_size"]), 
        padding='same', 
        activation=decoder_config["activation"], 
        strides=decoder_config["conv_t_1"]["stride"]
    )(x)
    print("shape after first convolutional transpose layer", x.shape)

    x = layers.Conv2DTranspose(
        decoder_config["conv_t_2"]["filter_num"], 
        tuple(decoder_config["conv_t_2"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_2"]["stride"],
        activation=decoder_config["activation"]
    )(x)
    print("shape after second convolutional layer", x.shape)

    x = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_3"]["filter_num"], 
        tuple(decoder_config["conv_t_3"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_3"]["stride"],
        activation=decoder_config["activation"]
    )(x)
    print("shape after second convolutional layer", x.shape)
    
    x_recon = keras.layers.Conv2DTranspose(
        decoder_config["conv_t_4"]["filter_num"], 
        tuple(decoder_config["conv_t_4"]["kernel_size"]), 
        padding='same', 
        strides=decoder_config["conv_t_4"]["stride"],
        activation=decoder_config["conv_t_4"]["activation"],
        name = 'reconst'
    )(x)
    print("shape after conv recon layer", x_recon.shape)

#     x_recon = keras.layers.Cropping2D(cropping=(1, 0))(x_recon)
#     print("shape after cropping", x_recon.shape)

    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_recon],name= 'reconst')

    return variational_decoder




def decoder_dense(
    original_input: tuple,
    decoder_config: dict, 
):
    """
    Decoder with fully connected layer 
    """
    decoder_inputs = keras.layers.Input(shape=[decoder_config["latent_dim"]])
    
    # x = keras.layers.Dense(np.prod(shape_before_flat[1:]), activation=decoder_config["activation"])(decoder_inputs)
    x = keras.layers.Dense(decoder_config['dense1']['unit_num'],kernel_initializer=initializer,
        activation=decoder_config["activation"],)(decoder_inputs)


    x = keras.layers.Dense(decoder_config['dense2']['unit_num'],kernel_initializer=initializer,
        activation=decoder_config["activation"],)(x)
    x = keras.layers.Dense(decoder_config['dense3']['unit_num'],kernel_initializer=initializer,
        activation=decoder_config["activation"])(x)

    x = keras.layers.Dense(decoder_config['dense4']['unit_num'],kernel_initializer=initializer,
        activation=decoder_config["activation"])(x)
    x = keras.layers.Dense(decoder_config['dense4']['unit_num'],kernel_initializer=initializer,
        activation=decoder_config["activation"])(x)
    x = keras.layers.Dense(decoder_config['dense5']['unit_num'],kernel_initializer=initializer)(x)
       
    
    print("shape at beginning of decoder", x.shape)

    x_recon = layers.Reshape(original_input,name = 'Reconst')(x)
    print("shape after conv recon layer", x_recon.shape)


    variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[x_recon],name= 'reconst')

    return variational_decoder


def encoder_dense(input_shape: tuple,input_shape_c: tuple, encoder_config: dict):
    """
    Encoder with fully connected layers
    """

    class EncoderResult():
        pass 

    encoder_result = EncoderResult()
    
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    inputs_c = tf.keras.layers.Input(shape=input_shape_c)
    

    x =layers.Flatten()(inputs)
    x = layers.Dense(
        encoder_config["dense_1"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"], 
    )(x)

    x = layers.Dense(
        encoder_config["dense_2"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"], 
    )(x)

    x = layers.Dense(
        encoder_config["dense_3"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"], 
    )(x)

    x = layers.Dense(
        encoder_config["dense_4"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"] 
    )(x)
    
    xc = layers.Concatenate(axis = -1)([x,inputs_c])
    xc = layers.Dense(
        encoder_config["dense_5"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"] 
    )(xc)
    
    xc = layers.Dense(
        encoder_config["dense_6"]["unit_num"], kernel_initializer=initializer,
        activation=encoder_config["activation"] 
    )(xc)


    x_out = layers.Dense(encoder_config["latent_dim"],encoder_config["dense_7"]["activation"], name='xmean')(xc)
    print("shape o output", x_out.shape)

    # Instantiate Keras model for VAE encoder 
    vae_encoder = Model(inputs = [inputs,inputs_c], outputs=[x_out])

    # Package up everything for the encoder
    encoder_result.inputs = inputs
    encoder_result.inputs_c = inputs_c
    encoder_result.x = x
    encoder_result.vae_encoder = vae_encoder 

    return encoder_result

def dense_gen(input_shape: tuple, dense_config: dict):
    """
    Fully connected feedforwad neural network
    """
    class denseResult():
        pass 

    dense_result = denseResult()
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense (dense_config["layer_1"]["unit_num"], 
        activation=dense_config["activation"])(inputs)

    x = layers.Dense (dense_config["layer_2"]["unit_num"], 
        activation=dense_config["activation"])(x)

        
    x = layers.Dense (dense_config["layer_3"]["unit_num"], 
        activation=dense_config["activation"])(x)
    
    x = layers.Dense (dense_config["layer_4"]["unit_num"], 
        activation=dense_config["activation"])(x)

    x_out = layers.Dense (dense_config["layer_5"]["unit_num"], 
        activation=dense_config['layer_5']['activation'],name= 'precip')(x)
    print ('output shape is: ',x_out.shape)
    
    dense_nn = Model(inputs = [inputs], outputs=[x_out],name = 'precip')
    dense_result.inputs = inputs
    dense_result.dense_nn = dense_nn

    
    return dense_result


def cloud_model(input_shape: tuple, dense_config: dict):
    """
    Fully connected feedforwad neural network
    """
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense (dense_config["layer_1"]["unit_num"], 
        activation=dense_config["activation"])(inputs)

    x = layers.Dense (dense_config["layer_2"]["unit_num"], 
        activation=dense_config["activation"])(x)

    x_out = layers.Dense (dense_config["layer_4"]["unit_num"],name = 'cloud')(x)
        
    print ('output shape is: ',x_out.shape)
    
    model_dense = Model(inputs = [inputs], outputs=[x_out],name='cloud')
    
    return model_dense

    
    

