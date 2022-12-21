import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib import cm
from tqdm import tqdm
import pickle
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import r2_score
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,Conv1D, MaxPooling1D,Input,Lambda,LeakyReLU,Reshape
from keras.models import Model
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
