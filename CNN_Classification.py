import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Dataset_Transformed.csv', header=0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
X = dataset.iloc[:, 0:42].values
Y = dataset.loc[:, ['attack_cat']].values
#Y = dataset.iloc[:, 43].values
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.2)
#print(train_X.head())


print('Training data shape : ', train_X.shape, train_Y.shape)

#y=dataset.attack_cat
#x=data.drop('temp',axis=1)



print("target column length in training", len(train_Y))
classes = np.unique(train_Y)
print("Encoded values for targets in training", classes)
nClasses = len(classes)
print('Total number of classes in training: ', nClasses)

#print(train_X[0].shape)
#print(train_Y[0].shape)
train_X = train_X.reshape(-1, 14, 3, 1)
print("New shape of the training data class labels", train_X.shape)
"""dataset = pd.read_csv('Testing_Transformed.csv', header=0)
test_X = dataset.iloc[:, 0:42].values
test_Y = dataset.iloc[:, 43].values"""
#print('Output classes for testing: ', classes)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Find the unique numbers from the train labels
classes = np.unique(test_Y)
nClasses = len(classes)
print('Total number of classes for testing: ', nClasses)
print("Shape of target label", test_Y.shape)
print('Output classes for testing: ', classes)
#Required for Plotting
test_y = test_Y.flatten()
#print(test_X[0].shape)
test_X = test_X.reshape(-1, 14, 3, 1)
print("New shape of the testing data class labels", test_X.shape)

# Change the labels from categorical to one-hot encoding
train_Y=train_Y.flatten()
test_Y=test_Y.flatten()
train_Y = pd.get_dummies(train_Y)
test_Y = pd.get_dummies(test_Y)
print('one-hot')
print(train_Y)
print(test_Y)

"""
# Display the change for category label using one-hot encoding
classes = np.unique(train_Y)
print("one hot Encoded values for targets in training", classes)
nClasses = len(classes)
print('Total number of outputs in training after one-hot encoding: ', nClasses)
classes = np.unique(test_Y)
print("one hot Encoded values for targets in testing", classes)
nClasses = len(classes)
print('Total number of outputs in testing after one-hot encoding: ', nClasses)

#print("First row of the one-hot encoded target column in the training dataset is ",  train_Y)

"""
#MODEL -->

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
batch_size = 256
epochs = 200
num_classes = 10
netmod = Sequential()
netmod.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(14, 3, 1),padding='same'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(MaxPooling2D((2, 2),padding='same'))
netmod.add(Dropout(0.25))
netmod.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
netmod.add(Dropout(0.25))
netmod.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
netmod.add(Dropout(0.25))
netmod.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
netmod.add(Dropout(0.30))
netmod.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
netmod.add(Dropout(0.50))
netmod.add(Flatten())
netmod.add(Dense(128, activation='linear'))
netmod.add(LeakyReLU(alpha=0.1))
netmod.add(Dropout(0.40))
netmod.add(Dense(num_classes, activation='softmax'))
netmod.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(netmod.summary())



#netmod.save("netmod_dropout.h5py")
#Loading Model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#netmod = model_from_json(loaded_model_json)
# load weights into new model
netmod.load_weights("model.h5")
print("Loaded model from disk")
#netmod_dropout = netmod.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1)
test_eval = netmod.evaluate(test_X, test_Y, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
'''model_json = netmod.to_json()
with open("model.json", "w") as json_file:
   json_file.write(model_json)
#serialize weights to HDF5
netmod.save_weights("model.h5")
#print("Saved model to disk")'''

#subtracting one from each element of list
test_y[:]=[x-1 for x in test_y]

pred_Y=netmod.predict(test_X)
print("Prediction Y")
print(pred_Y.argmax(axis=1)[0:100])
print("Test Y")
print(test_y[0:100])
cm = confusion_matrix(test_y,pred_Y.argmax(axis=1))
cr=classification_report(test_y, pred_Y.argmax(axis=1))
print("Confusion Matrix")
print(cm)
print("Classification Report")
print(cr)

#PLOTTING -->
accuracy = netmod_dropout.history['acc']
#val_accuracy = netmod_dropout.history['val_acc']
loss = netmod_dropout.history['loss']
#val_loss = netmod_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy')

plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
plt.legend()
plt.show()

