#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:49:12 2021

@author: catcry

                    >> nsl4conf Main File
"""
from nsl4conf.Models.model_gen import *

##############################################################################
##                                                                          ##
##                           Variable Definition                            ##                                                     
##                                                                          ##
##############################################################################

#======>>     Sets
faddr = "/home/catcry/cc/nsl4conf/sets/iris.data"
seperator = ','
data = pd.read_csv(faddr,sep = seperator)
Xtrain = 
Ytrain = 

Xvalid = 
Yvalid = 

Xtest = 
Ytest = 

#------------------------------------------------------------------------------
#=====>>     model struction parameters
architecture = np.array([100,200,400,800,1600,800,400,200,100,32])
activation = 'relu'
bias = True

#------------------------------------------------------------------------------
#=====>>     Optimizer
optimizer_type = 'adam'
learning_rate = 
decay_rate = 0

#------------------------------------------------------------------------------
#=====>>     model_fit parameters
num_epochs = 100
batch_size = 200
class_weight = None
validation_split = 0
validation_data = (Xvalid,Yvalid)
verbose = 1
shuffle = False

#------------------------------------------------------------------------------
#======>>     complie parmeters
loss_function = 'mse'
metrics = ['mae', 'acc']



#------------------------------------------------------------------------------



##############################################################################
##                                                                          ##
##                           Model Implementation                           ##                                                     
##                                                                          ##
##############################################################################

net_model = Model_gen(net_type = "Deep")

net_model.cc_optimizer(learning_rate = learning_rate, decay = 0, optimizer = optimizer_type)

net_model.cc_model(net_input = Xtrain, architecture = architecture,\
                  optimizer= net_model.cc_optimizer , activation = activation)

net_model.compile (loss=loss_function, \
                            optimizer = net_model.cc_model.optimizer,\
                            metrics=metrics)


history_g = model_g.fit (Xtrain,Ytrain, \
                                 epochs = num_epochs, \
                                 batch_size = batch_size, \
                                 class_weight = class_weights,\
                                 validation_split = validation_split,\
                                 validation_data = validation_data ,\
                                 # validation_data = (Xtest_21,Ytest_21_1hot),\
                                 verbose = verbose, \
                                 shuffle = shuffle,\
                                 # callbacks=[globals()['csv_logger_U'+str(num_layers[0])]]
                                 )