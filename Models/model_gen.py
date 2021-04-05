# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:05:06 2019

@author: catcry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as k

#Keras Lib Essens:
#import tensorflow.python.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM ,GRU, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D,  AveragePooling1D
#from tensorflow.keras.layers import CuDNNLSTM
#from tensorflow.keras.layers.advanced_activations import PReLU,LeakyReLU,ELU, ThresholdedReLU, ReLU
from tensorflow.keras import losses , optimizers, regularizers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model

#Ski-Learn Essentials
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#ski-learn ML
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


#from funz import num_of
#from funz import rnd

from nsl4conf.Preprocess.preparation import prep_stand, prep_manual, prepstand_multi

# from prepstand import prep_stand
from prepstand_multi import prep_stand_multi
# from prepmanual import prep_manual

class Deep_gen():
    """ This class is used to generate Deep models,
    for now : A model that is an object of this class can call two methods:
        1. 
    """
    
    def __init__(self, net_type, **options):
        # self.arch = architecute
        # self.opt = optimizer
        self.net_type = net_type
        
    
    def cc_model(self,net_input,architecture, optimizer='adam',\
                  activation = 'elu',bias = True, **options):
        self.input = net_input
        self.arch = architecture
        self.input_shape = net_input.shape
        if type(activation) == str:
            self.act = [activation for layer in self.arch]
            
        elif type(activation) == list and len(activation)!= len(self.arch):
            print ('The length of activation list is not equal to the no. of layers')
            print ('The default activation function will be used : ("elu")')
            self.act = ['elu' for layer in self.arch]
            
            
        
        self.model = Sequential()
        self.model.add(Dense(self.arch[0], activation = self.act[0],\
                        input_shape = (self.input_shape[1],self.input_shape[2]),\
                        use_bias = bias))
        
        for layer_no in range(1,len(self.arch)):
            self.model.add(Dense(self.arch[layer_no], \
                                 activation = self.act[layer_no],\
                                 use_bias = bias))
         
        return self.model       
                
                
        
        
        def cc_optimizer(self, learning_rate,decay_rate = 0, optimizer = 'adam'):
            
                        
            if optimizer =='sgd':                        
                self.cc_optimizer = optimizers.SGD(lr=learning_rate,\
                                     decay = decay_rate, \
                                     momentum = moment, \
                                     nesterov=True)
                
            elif optimizer =='rms':
                #--------------------------------------------------------------
                self.cc_optimizer = optimizers.RMSprop(lr = learning_rate, \
                                         rho= 0.9, \
                                         epsilon = None,\
                                         decay = decay_rate)
                    
            elif optimizer =='adagrad':         
                #--------------------------------------------------------------
                self.cc_optimizer = optimizers.Adagrad (lr = learning_rate , \
                                              epsilon = None , \
                                              decay = decay_rate)
                
            elif optimizer =='adadelta':                    
                #--------------------------------------------------------------
                self.cc_optimizer = optimizers.Adadelta(lr = learning_rate, \
                                         rho=0.95 , \
                                         epsilon = None,\
                                         decay = decay_rate)
                
            
        
            elif optimizer =='nadam':                
                self.cc_optimizer = optimizers.Nadam(lr = learning_rate, \
                                         beta_1 = 0.9, \
                                         beta_2 = 0.999, \
                                         epsilon = None, \
                                         schedule_decay = 0.004)
                    
            else: 
                self.cc_optimizer = optimizers.Adam(lr = learning_rate, \
                                         beta_1 = 0.9 , \
                                         beta_2 = 0.999 , \
                                         epsilon = None,\
                                         decay = decay_rate,\
                                         amsgrad = True )
        
            return self.cc_optimizer
        
        
        
        
        
        
        
        
        
        