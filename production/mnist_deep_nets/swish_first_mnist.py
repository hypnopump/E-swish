""" Repeating MNIST first experiment of swish paper. """

import gc # Garbage collector
import logging
import numpy as np

# Record settings
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="e_swish_2_log.txt",format = LOG_FORMAT, level = logging.DEBUG, filemode = "a")
logs = logging.getLogger()

np.random.seed(2)

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# For adding new activation function
from keras import backend as K
from keras.datasets import mnist
from keras.utils.generic_utils import get_custom_objects
from keras.utils import np_utils

print("Modules imported")

# Data retrieval and preprocess
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", Y_train.shape)
# Normalization and reformatting of labels
nb_classes = 10
X_train = X_train / 255.0
X_test = X_test / 255.0
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, 
                                test_size = 0.1, random_state=random_seed)

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.reshape(-1,784)
X_val = X_val.reshape(-1,784)
X_test = X_test.reshape(-1,784)
# test = test.values.reshape(-1,28,28,1)
print("Data preprocessed", X_train.shape, X_val.shape, X_test.shape)


# Define custom activations
def swish(x):
    return x*K.sigmoid(x)

def e_swish_2(x):
    sigmoid = K.sigmoid(x)
    return K.maximum(x*sigmoid, x*(2-sigmoid))


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
def create(act, n):
    model = Sequential()
    # First conv block
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation(act))
    for i in range(n-1):
        if i%2 == 0:
            model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(Activation(act))
        model.add(Dropout(0.3))
        
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    
    return model


# Accuracy calculator helper
def accuracy(y_pred, y_test):
    y_hat = np.argmax(y_pred, axis=1)
    y = np.argmax(y_test, axis=1)

    good = np.sum(np.equal(y, y_hat))
    return float(good/len(y_test))


# E-swish-2 block
act = e_swish_2

logs_e_swish_2 = []
record_e_swish_2 = []
for n in range(23,42,3):
    ensembler = 0
    logger = [n]
    logs.info("\n \n Starting round with {0} layers".format(n))
    print("\n \n Starting round with {0} layers".format(n))
    for i in range(3):
        # Garbage collector
        gc.collect()
        # Set optimizer
        opt = SGD(lr=0.01, momentum=0.9)
        # Set callbacks (learning rate reducer and early stopping)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.35, min_lr=0.00001)
        early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose = 1)
        # Common params 
        epochs = 15
        batch_size = 128
        # Create and compile the model
        model = create(act, n)
        # Compile the model
        model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["accuracy"])
        # Train the model
        history = model.fit(X_train,Y_train, epochs = epochs, validation_data = (X_val,Y_val),
                            verbose = 1 , callbacks=[learning_rate_reduction, early_stop])
        
        # Record accuracy of each model and save it
        logger.append(model.evaluate(X_test, Y_test)[1])
        logs.info("Accuracy "+str(i)+": "+str(logger[-1]))
        # Calculate probabilities of test data and sum them toghether
        ensembler += model.predict_proba(X_test)
        # Clear session (GPU MEMORY)
        K.get_session().close()
        K.set_session(K.tf.Session())
        del model, history, learning_rate_reduction, early_stop, opt
     
    # Calculate the median accuracy
    ensembled = accuracy(ensembler, Y_test)
    # Save the ensembled accuracy and the three models accuracy
    record_e_swish_2.append([n, ensembled])
    logs_e_swish_2.append(logger)
    del ensembler, ensembled
    logs.info("Ensembled accuracy: "+str(record_e_swish_2[-1]))
    logs.info("Logs: +"+str(logs_e_swish_2[-1]))
    
logs.info("\n \n \n")
logs.info("Logs e_swish_2: "+str(logs_e_swish_2))
logs.info("Record e_swish_2: "+str(record_e_swish_2))


# # Swish block
# act = swish

# logs_swish = []
# record_swish = []
# for n in range(23,42,3):
#     ensembler = 0
#     logger = [n]
#     print("\n \n Starting round with {0} layers".format(n))
#     for i in range(3):
#         # Garbage collector
#         gc.collect()
#         # Set optimizer
#         opt = SGD(lr=0.01, momentum=0.9)
#         # Set callbacks (learning rate reducer and early stopping)
#         learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.35, min_lr=0.00001)
#         early_stop = EarlyStopping(monitor='val_acc', patience=5, verbose = 1)
#         # Common params 
#         epochs = 15
#         batch_size = 128
#         # Create and compile the model
#         model = create(act, n)
#         # Compile the model
#         model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["accuracy"])
#         # Train the model
#         history = model.fit(X_train,Y_train, epochs = epochs, validation_data = (X_val,Y_val),
#                             verbose = 1 , callbacks=[learning_rate_reduction, early_stop])
        
#         # Record accuracy of each model and save it
#         logger.append(model.evaluate(X_test, Y_test)[1])
#         # Calculate probabilities of test data and sum them toghether
#         ensembler += model.predict_proba(X_test)
#         # Clear session (GPU MEMORY)
#         K.get_session().close()
#         K.set_session(K.tf.Session())
#         del model, history, learning_rate_reduction, early_stop, opt
     
#     # Calculate the median accuracy
#     ensembled = accuracy(ensembler, Y_test)
#     print(ensembled)
#     # Save the ensembled accuracy and the three models accuracy
#     record_swish.append([n, ensembled])
#     logs_swish.append(logger)
#     del ensembler, ensembled
    
# print("\n \n \n")
# print("Logs swish: ", logs_swish)
# print("Record swish: ", record_swish)