# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 03:08:34 2017




"""


from __future__ import print_function
import numpy as np
np.random.seed(1337)#it will generate the same random number array

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils  import np_utils
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D

from keras import backend as K

K.set_image_dim_ordering('th')
#K.image_dim_ordering() # 'th'


aaa=np.load("indiareadorigin10class_200onevector_normalize.npz")#read from file
                   
data=aaa["arr_0"]

label=aaa["arr_1"]

Dtrain=np.empty((1600,1,200),dtype="float32")
Ltrain=np.empty((1600),dtype="int32")
Dtrain1=np.empty((1600,200),dtype="float32")


Dtrain=data[:1600,:,:]
Ltrain=label[:1600]
Dtrain1=Dtrain.reshape([1600,200,1])


Dtest=np.empty((400,1,200),dtype="float32")
Ltest=np.empty((400),dtype="int32")
Dtest1=np.empty((400,200),dtype="float32")


Dtest=data[1600:,:,:]
Ltest=label[1600:]
Dtest1=Dtest.reshape([400,200,1])

Dtrain = Dtrain.astype('float32')
Dtest = Dtest.astype('float32')
#Dtrain /= 255
#Dtest /= 255

print(data.shape[0],' samples')
print(label.shape[0],'labels')



Ltrain = np_utils.to_categorical(Ltrain, 10)
Ltest = np_utils.to_categorical(Ltest, 10)




nb_filter=32
kernel_size=5
border_mode='valid'
activation='relu'




model = Sequential()

model.add(Conv1D(nb_filter=32,kernel_size=3,padding='valid',activation='relu',input_shape=(200,1)))
model.add(Activation('relu'))
model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.25))



model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])


hist=model.fit(Dtrain1, Ltrain, batch_size=100, nb_epoch=1000,
          verbose=1, validation_data=(Dtest1, Ltest))
          



model.save('20170320indiacnn-200-1D-godeep-keras2-5.h5')  # creates a HDF5 file 'my_model.h5'

model.save_weights('20170320indiacnn-200-1D-godeep-keras2-5weithts.h5')
   
score = model.evaluate(Dtest1, Ltest, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

#model.save('mymodel.h5')  # creates a HDF5 file 'my_model.h5'
#model.save_weights('mymodelweithts.h5')
#del model  # deletes the existing model