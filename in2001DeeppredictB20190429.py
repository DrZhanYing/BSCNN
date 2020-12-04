# -*- coding: utf-8 -*-
"""

"""

from __future__ import print_function
import numpy as np

from keras.utils  import np_utils
import random

import indiafunction
from keras.models import load_model
   



  
dictofclass={3:0,4:0,5:0,6:0,8:0,10:0,11:0,12:0,13:0,14:0}
labelofclass={3:0,4:1,5:2,6:3,8:4,10:5,11:6,12:7,13:8,14:9}



dataAll,labelOriAll,labelAll=indiafunction.getalldataLabel("indiareadorigin10class_200onevector_normalize.npz")



dataAll=dataAll.reshape([2000,200,1])


model = load_model('i200_1D_DeepModel.h5')


indexofclass=8


dataofoneclass,nbofclass=indiafunction.getoneclassdatalabel(dataAll,labelOriAll,indexofclass)


m=50
d_selectnumber10_50=indiafunction.distansDensity(dataAll,10,m)

precurr=0.8
numberLoop=200


nbBSelectnowmax,dataBS,precurrnowmax,preproba = indiafunction.predictBandS(dataofoneclass,indexofclass,d_selectnumber10_50,model,precurr,numberLoop)


nbBSelectnowmax1,dataBS1,precurrnowmax1,preproba1 = indiafunction.predictBandRandom200(dataofoneclass,indexofclass,m,model,precurr,numberLoop)



    
    
















