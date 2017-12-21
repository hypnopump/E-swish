import keras
import keras.backend as K
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
import sklearn.metrics as metrics
import wide_residual_network as wrn
from keras.callbacks import CSVLogger
import logging

print("Imported modules")


# Data retrieval and preprocess
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

print("Data retrieved and preprocessed")


# Custom activation function 2
# positive part of swish mirrored across x=1
def e_swish_2(x):
    return K.maximum(x*K.sigmoid(x), x*(2-K.sigmoid(x)))

act, act_name = e_swish_2, "e_swish_2"


# data augmentation - no rotation range as described in the paper
datagen = ImageDataGenerator(
#     rotation_range=15,
    width_shift_range=0.125,
    height_shift_range=0.125,
    horizontal_flip=True, 
    fill_mode="reflect"
    )
datagen.fit(x_train)

print("Dta augmentation API fitted")


# Instantiate the model
init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
model = wrn.create_wide_residual_network(act, init_shape, nb_classes=10, N=3, k=4, dropout=0.00)

print("Model created and ready")


# Train the model
batch_size  = 128
epochs = 30

# CSV LOGGER CALLBACK + LOGGING SETTINGS
name = 'NASNet-CIFAR-10_n1_1step'
csv_logger = CSVLogger(name+'.csv')
# Record settings
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename=name+'.txt',format = LOG_FORMAT, level = logging.DEBUG, filemode = "a")
logs = logging.getLogger()

opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("Finished compiling")

####################
# Network training #
####################
                     
print("Training up to 600 epochs")
for i in range(2):
    his = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,
    						epochs=epochs,verbose=1,validation_data=(x_test,y_test), callbacks=[csv_logger])
    model.save('wrn_16_4_01_'+str(i)+'.h5')
    logs.info(str(his.history))
    logs.info("\n")