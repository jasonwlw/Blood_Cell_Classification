import random
import numpy as np
import keras
import os
import cv2
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from time import time
from PIL import ImageEnhance,Image
from statistics import mode

#--------SET SEEDS--------#
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
#-------SET SEEDS--------#

class MLP_Cells:

    def __init__(self,layers = (100,200,100),alpha = 0.001,num_epochs = 1000,batch_size = 50,dropout = 0.2,
                 input_dim = 120*120*3,num_classes=4,normalize = True,SEED=42,balanceb=False,resample=True,save_every=None,
                 individual=None):
        self.layers = layers
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalize = normalize
        self.SEED = SEED
        self.balanceb = balanceb
        self.resample = resample
        self.save_every = save_every
        self.individual = individual

        self.im_list = []
        self.y_list = []

        self.initialize_model()

        ###OneHotEncoder,LabelEncoder for y variables
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()


    def minmax_normalization(self,im):
        im = im / 255.
        return np.asarray(im)

    def get_array_shapes(self):
        ### use median shapes, max shapes, something like that
        pass

    def resize_im(self,im):
        ### from jdhao #maybe not
        desired_size = 120

        old_size = im.shape[:2]
        # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (desired_size, desired_size))
        return im


    def read_data(self,datapath='../train/',test_size = 0.2):
        self.dir = datapath
        for fil in os.listdir(datapath):
            if fil.endswith('.png'):
                self.im_list.append(fil)
            elif fil.endswith('.py'):
                pass
            else:
                self.y_list.append(fil)
        self.align_data()
        X_train,X_test,y_train,y_test = train_test_split(self.im_list,self.y_list,test_size=test_size,random_state=42)
        return X_train,X_test,y_train,y_test

    def target_read(self,filename):
        with open(filename,'r') as f0:
            return f0.readlines()[0] ### readlines() returns an array; to get the string itself we have to index it


    def align_data(self):
        ### Align data and target based on the number in the filename
        im_order = [int(im.split('_')[1].split('.')[0]) for im in self.im_list]
        y_order = [int(y.split('_')[1].split('.')[0]) for y in self.y_list]
        im_list_ord = []
        for im_ind,im in sorted(zip(im_order,self.im_list)):
            im_list_ord.append(im)
        self.im_list = im_list_ord

        y_list_ord = []
        for y_ind,y in sorted(zip(y_order,self.y_list)):
            y_list_ord.append(y)
        self.y_list = y_list_ord

    def initialize_model(self):
        self.initialize_input_layer()
        self.initialize_middle_layers()
        self.initialize_final_layer()
        self.compile_model()

    def initialize_input_layer(self):
        ### build first keras layer; images are 101X100X3 so input dim =
        if type(self.dropout) == type(np.asarray([])) or type(self.dropout) == type([]):
            self.model = Sequential([
                Dense(self.layers[0], input_dim=self.input_dim, kernel_initializer=weight_init),
                Activation('relu'),
                Dropout(self.dropout[i]),
                BatchNormalization()
            ])
        else:
            self.model = Sequential([
                Dense(self.layers[0], input_dim=self.input_dim, kernel_initializer=weight_init),
                Activation('relu'),
                Dropout(self.dropout),
                BatchNormalization()
            ])

    def initialize_middle_layers(self):
        for i in range(1,len(self.layers)):
            self.model.add(Dense(self.layers[i],activation='relu',kernel_initializer=weight_init))
            if type(self.dropout) == type(np.asarray([])):
                self.model.add(Dropout(self.dropout[i]))
            else:
                self.model.add(Dropout(self.dropout))
            self.model.add(BatchNormalization())

    def initialize_final_layer(self):
        self.model.add(Dense(self.num_classes,activation='softmax',kernel_initializer=weight_init))

    ### Figure out cohens kappa and macro-averaged f1 score metrics
    def compile_model(self):
        self.model.compile(optimizer=Adam(lr=self.alpha),loss='categorical_crossentropy',metrics=['accuracy'])

    def rebalance_batch(self,X_train,y_train):
        self.training_generator = BalancedBatchGenerator(X_train,y_train.todense(),sampler=RandomOverSampler(),batch_size=self.batch_size,random_state=self.SEED)

    def data_preparation(self,X_train,X_test,y_train,y_test):
        X_train = [self.dir+x for x in X_train]
        X_test = [self.dir+x for x in X_test]
        y_train = [self.dir+x for x in y_train]
        y_test = [self.dir+x for x in y_test]

        ### Wait until augmentation is done to make X_train a numpy array
        X_train = list(map(cv2.imread, X_train))
        X_test = np.asarray(list(map(cv2.imread, X_test)))


        ### Have to perform one hot encoding on target
        y_train = list(map(self.target_read, y_train))

        #class_repeats = self.get_target_counts(y_train)
        X_train,y_train = self.augmenter(X_train,y_train,class_repeats)


        self.le = self.le.fit(["red blood cell", "ring","schizont", "trophozoite"])
        y_train = self.le.transform(np.asarray(y_train).reshape(-1,1))
        y_test = list(map(self.target_read, y_test))
        y_test = self.le.transform(np.asarray(y_test).reshape(-1,1))
        ### instead of taking up time in the fit function, maybe move

        if self.normalize:
            X_train = np.asarray(list(map(self.minmax_normalization, X_train)))
            X_test = np.asarray(list(map(self.minmax_normalization, X_test)))


        X_train = np.asarray(list(map(self.resize_im,X_train)))
        X_test = np.asarray(list(map(self.resize_im,X_test)))

        X_train = np.asarray(list(map(np.ndarray.ravel,X_train)))
        X_test = np.asarray(list(map(np.ndarray.ravel,X_test)))


        if self.resample:
            X_train,y_train = RandomOverSampler().fit_resample(X_train,y_train)

        if self.individual is not None:
            y_train = np.array(list(map(self.filter,y_train)))
            y_test = np.array(list(map(self.filter,y_test)))
            X_train,y_train = RandomUnderSampler().fit_resample(X_train,y_train)
        print(len(X_train))


        y_train = self.ohe.fit_transform(np.asarray(y_train).reshape(-1, 1))
        y_test = self.ohe.transform(np.asarray(y_test).reshape(-1, 1))

        return X_train,X_test,y_train,y_test

    def filter(self,arr_ele):
        ### Only used for self.individual method
        if arr_ele == self.individual:
            return 1
        else:
            return 0

    def get_target_counts(self,y_train):
        ring = y_train.count('red blood cell') // y_train.count('ring')

        trophozoite = y_train.count('red blood cell') // y_train.count('trophozoite')
        schizont = y_train.count('red blood cell') // y_train.count('schizont')
        return {'ring':ring,'trophozoite': trophozoite,'schizont':schizont,'red blood cell':2}

    def augmenter(self,X_train,y_train,class_repeats):
        print("Augmenting...")
        rot = {0:0,1:cv2.ROTATE_90_CLOCKWISE,2:cv2.ROTATE_180,3:cv2.ROTATE_90_COUNTERCLOCKWISE}
        bright = np.arange(0.9,1.15,0.05)
        for i in range(len(X_train)):
            for j in range(class_repeats[y_train[i]]-2):
                X = self.apply_random_transform(X_train[i],rot,bright)
                X_train.append(X)
                y_train.append(y_train[i])
        return X_train,y_train

    def apply_random_transform(self,X,rot,bright):
        pick_rot = np.random.randint(0,4)
        pick_bright = np.random.randint(0,len(bright))
        if rot[pick_rot] == 0:
            im = Image.fromarray(X)
            im = ImageEnhance.Brightness(im)
            im = im.enhance(bright[pick_bright])
            return np.array(im)
        else:
            X = cv2.rotate(X,rot[pick_rot])
            im = Image.fromarray(X)
            ### All PIL enhancements below
            im = ImageEnhance.Brightness(im)
            im = im.enhance(bright[pick_bright])
            return np.array(im)


    def view_image(self,img):
        ### check images
        plt.plot(img)
        plt.show()

    def fit(self):
        X_train,X_test,y_train,y_test = self.read_data(datapath='../train/')
        X_train,X_test,y_train,y_test = self.data_preparation(X_train,X_test,y_train,y_test)
        if not self.balanceb:
            if self.save_every == None:
                self.model.fit(X_train,y_train,epochs=self.num_epochs,validation_data=(X_test,y_test),verbose=1)
            else:
                if self.individual is None:

                    checkpoints = keras.callbacks.ModelCheckpoint('mlp_witryjw_ind_{epoch:02d}-{val_loss:.2f}'+str(self.layers)+'.hdf5',
                                                              monitor='val_loss',verbose=1,save_weights_only=False,
                                                              period=self.save_every)
                else:
                    checkpoints = keras.callbacks.ModelCheckpoint('mlp_witryjw_ind'+str(self.individual)+'_{epoch:02d}-{val_loss:.2f}'+str(self.layers)+'.hdf5',
                                                              monitor='val_loss',verbose=1,save_weights_only=False,
                                                              period=self.save_every)
                self.model.fit(X_train,y_train,epochs=self.num_epochs,validation_data=(X_test,y_test),
                               verbose=1,callbacks=[checkpoints])
        else:
            self.rebalance_batch()
            self.model.fit_generator(generator=self.training_generator,epochs=self.num_epochs,validation_data=(X_test,y_test))
        return X_test,self.ohe.inverse_transform(y_test)

    def predict(self,X_test):
        return np.argmax(self.model.predict(X_test),axis=1)


params = {'layers': (100,200,200,100),
          'alpha': 5e-7,
          'batch_size': 32,
          'num_epochs': 600}


model_params = []
t0 = time()
i = 18

modeli = MLP_Cells(num_epochs=params['num_epochs'], alpha=params['alpha'], batch_size=params['batch_size'],
                   layers=params['layers'], save_every=50)

X_test, y_test = modeli.fit()

f1 = f1_score(y_test, modeli.predict(X_test), average='macro')
cohen = cohen_kappa_score(y_test, modeli.predict(X_test))
print(f1, cohen)

with open('./models/model.txt', 'a') as logfile:
    logfile.write('RUNTIME: ' + str(time() - t0))
    logfile.write('\n')
    logfile.write('f1 Score: ' + str(f1))
    logfile.write('\n')
    logfile.write('cohen Score: ' + str(cohen))
    logfile.write('\n')
    for key in params:
        logfile.write(key + ': ' + str(params[key]))
        logfile.write('\n')
    logfile.write('--------')