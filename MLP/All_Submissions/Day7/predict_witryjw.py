# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model

import os
os.system("sudo pip install opencv2-python")

import cv2



def resize_im(im):
    size = 120
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
    models = ['mlp_witryjw0.hdf5','mlp_witryjw1.hdf5','mlp_witryjw2.hdf5','mlp_witryjw3.hdf5','mlp_witryjw4.hdf5']
    model0 = load_model(models[0])
    model1 = load_model(models[1])
    model2 = load_model(models[2])
    model3 = load_model(models[3])
    model4 = load_model(models[4])
    y_pred = model1.predict(x)
    y_pred = y_pred[:, 1].reshape(-1, 1)
    for i in range(2, len(models)):
        model_cur = load_model(models[i])
        pred = model_cur.predict(x)
        # y_pred[:,i-1] *= pred[:,1]
        y_pred = np.hstack((y_pred, pred[:, 1].reshape(-1, 1)))

    y_pred0 = model0.predict(x)

    y_pred *= y_pred0
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5", etc.
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred, model0, model1, model2, model3, model4
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.