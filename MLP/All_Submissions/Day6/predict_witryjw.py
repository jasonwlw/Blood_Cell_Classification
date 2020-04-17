# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model

import os
os.system("sudo pip install opencv-python")

import cv2



def resize_im(im):
    size = 100
    im = cv2.resize(im, (size, size))
    return im

def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    ### make list of paths into list of numpy arrays
    x = np.asarray(list(map(cv2.imread,x)))
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    #x = x.reshape(len(x), -1)
    x = x / 255
    # Write any data prep you used during training
    x = np.asarray(list(map(resize_im, x)))
    x = np.asarray(list(map(np.ndarray.flatten, x)))
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('./mlp_witryjw.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.

