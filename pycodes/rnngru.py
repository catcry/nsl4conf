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


from funz import num_of
from funz import rnd
from prepstand import prep_stand
from prepstand_multi import prep_stand_multi
from prepmanual import prep_manual

#=================
num_classes = 1 #||
#=================

##=======>> Makin Reproducible Results   <====
#import random as rn
#import os 
#
#os.environ['PYTHONHAHSEED'] = '0'
#np.random.seed(20)
#rn.seed(50)
#tf.random.set_seed(100)
#session_conf =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
#sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config = session_conf) 
#k.set_session(sess)
#


#==============================================================================
#========================>  Regular Data Prepare   <===========================
#==============================================================================

faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTrain+.csv'
seperator =  ','
[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest,Ztrain,Zvalid,Ztest] = \
                                prep_stand (faddr,seperator,1,10,0,0,1)

faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTest+.csv'
[Xtra,Ytra,Xval,Yval,Xtest,Ytest,Ztra,Zval,Ztest] = prep_stand (faddr,seperator,0,0,1,100,1)

faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTest-21.csv'
[Xtra,Ytra,Xval,Yval,Xtest_21,Ytest_21,Ztra,Zval,Ztest21] = prep_stand (faddr,seperator,0,0,1,100,1)


Xtrain = Xtrain.reshape((len(Xtrain),1,41))
Xvalid = Xvalid.reshape((len(Xvalid),1,41))
Xtest  = Xtest.reshape((len(Xtest),1,41))
Xtest_21 = Xtest_21.reshape((len(Xtest_21),1,41))

Ytrain_1hot = np.zeros([len(Ytrain),num_classes])
Yvalid_1hot = np.zeros([len(Yvalid),num_classes])
Ytest_1hot = np.zeros([len(Ytest),num_classes])
Ytest_21_1hot = np.zeros([len(Ytest_21),num_classes])

if num_classes != 1:
    for i in range(len(Ytrain)):
            Ytrain_1hot[i,Ytrain[i]] = 1
     
    for i in range(len(Yvalid)):
            Yvalid_1hot[i,Yvalid[i]] = 1
        
    for i in range(len(Ytest)):
            Ytest_1hot[i,Ytest[i]] = 1  
            
    for i in range(len(Ytest_21)):
            Ytest_21_1hot[i,Ytest_21[i]] = 1  

else:
      Ytrain_1hot = Ytrain
      Yvalid_1hot = Yvalid
      Ytest_1hot = Ytest
#___________________________________

Xtrain_scikit = Xtrain.reshape((len(Xtrain),41))
Xvalid_scikit = Xvalid.reshape((len(Xvalid),41))
Xtest_scikit  = Xtest.reshape((len(Xtest),41))
Xtest_21_scikit = Xtest_21.reshape((len(Xtest_21),41))

    
#______________________________________________________________________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#==============================================================================
#========================>  Manual minus (15 to 21)   <========================
#==============================================================================

faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\traintest.csv'
seperator =  ','

##prep_manual(faddr,seperator,\
#             vald,vald_percentage,\
#             test,test_percentage,\
##           hard_test,hard_test_level) :             

[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest,Xhard,Yhard] = prep_manual (faddr,seperator,1,10,1,20,0,0)

Xtrain = Xtrain.reshape((len(Xtrain),1,41))
Xvalid = Xvalid.reshape((len(Xvalid),1,41))
Xtest  = Xtest.reshape((len(Xtest),1,41))
#Xhard = Xhard.reshape((len(Xhard),1,41))

Ytrain_1hot = np.zeros([len(Ytrain),num_classes])
Yvalid_1hot = np.zeros([len(Yvalid),num_classes])
Ytest_1hot = np.zeros([len(Ytest),num_classes])
#Yhard_1hot = np.zeros([len(Yhard),num_classes])

if num_classes != 1:
    for i in range(len(Ytrain)):
            Ytrain_1hot[i,Ytrain[i]] = 1
    
    for i in range(len(Yvalid)):
            Yvalid_1hot[i,Yvalid[i]] = 1
        
    for i in range(len(Ytest)):
            Ytest_1hot[i,Ytest[i]] = 1  
    
    for i in range(len(Yhard)):
            Yhard_1hot[i,Yhard[i]] = 1  
else:
      Ytrain_1hot = Ytrain
      Yvalid_1hot = Yvalid
      Ytest_1hot = Ytest
#______________________________________________________________________________
#_____________________>    End of Manual Preparation    <______________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


                        
            
            #____________________________>   FOR 2-Class Mode    <_________________________
            #==============================================================================
            #==============================Getting Results Ready===========================
            #==============================================================================
            
            #ts_result = time.time()
            yh_Xtrain_g = model_g.predict(Xtrain)
            
            if num_classes == 1 :
                yh_Xtrain_g = np.around(yh_Xtrain_g,decimals=0).astype(np.int)
            else: 
                yh_Xtrain_g = yh_Xtrain_g.argmax(axis = 1)
               
               
            train_conf_matrix_g = confusion_matrix (Ytrain,yh_Xtrain_g)
            ac_train_g[i] = accuracy_score(Ytrain,yh_Xtrain_g)
            
            ac_train_g_mean_intime[i+1] = (sum(ac_train_g_mean_intime) + ac_train_g[i] ) / (i+1)        
            #______________________________________________________________________ 
            
            yh_valid_g = model_g.predict(Xvalid)
            
            if num_classes == 1:
                yh_valid_g = np.around(yh_valid_g,decimals=0).astype(np.int)
            else : 
                yh_test_g = yh_valid_g.argmax(axis = 1)
                
                
            valid_con_matrix_g = confusion_matrix (Yvalid , yh_valid_g)
            ac_valid_g[i] = accuracy_score(Yvalid_1hot , yh_valid_g)
            
            ac_test_g_mean_intime[i+1] = (sum(ac_valid_g_mean_intime) + ac_valid_g[i]) / (i+1)
           #______________________________________________________________________ 
            yh_test_g = model_g.predict(Xtest)
            
            if num_classes == 1:
                yh_test_g = np.around(yh_test_g,decimals=0).astype(np.int)
            else : 
                yh_test_g = yh_Xtrain_g.argmax(axis = 1)
                
                
            test_con_matrix_g = confusion_matrix (Ytest , yh_test_g)
            ac_test_g[i] = accuracy_score(Ytest_1hot , yh_test_g)
            
            ac_test_g_mean_intime[i+1] = (sum(ac_test_g_mean_intime) + ac_test_g[i]) / (i+1)
            #______________________________________________________________________
            
            
            
            #yh_hard = model_g.predict(Xhard)
            #yh_hard = np.around(yh_hard,decimals=0).astype(np.int)
            #hardtest_con_matrix =confusion_matrix (Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
            #ac_hard_test = accuracy_score(Yhard_1hot.argmax(axis=1),yh_hard.argmax(axis=1))
            
          
            
            yh_test_21_g = model_g.predict(Xtest_21)
            
            if num_classes == 1:
                yh_test_21_g  = np.around(yh_test_21_g,decimals=0).astype(np.int)
            else:
                yh_test_21_g = yh_test_21_g.argmax(axis=1)
                
            test_21_con_matrix_g =confusion_matrix (Ytest_21 , yh_test_21_g)
            ac_test_21_g[i] = accuracy_score(Ytest_21 , yh_test_21_g)
            
            ac_test_21_g_mean_intime[i+1] = ( sum(ac_test_21_g_mean_intime) + ac_test_21_g[i] )/(i+1)
            #______________________________________________________________________
           
            #----------------------------------------------------------------------
            
            
            print('------------------------------------------------------------------------')
            
            print("The Specification of Classifier is:")
            print('------------------------------------------------------------------------')
            print ("Layers Config : ")
            print ("Neural Network Type : ", nn_type)
            print ("No. of 1st Layer: ",num_layers[i], \
                   '|| No. of 2nd Layer :', num_2nd_layer,\
                   "|| No. of Classes: ",num_classes)
            print('------------------------------------------------------------------------')
            print ("Activtion Function: ",act_funct)
            print ("Output Layer Activtion Function: ",output_act_funct)
            #print('-----------------------------------------------------------------------')
            print ("Loss Function : " , loss_funct)
            print ("No. of Epochs : " , num_epochs)
            print('------------------------------------------------------------------------')
            print("Optimizer Config : ")
            print("Type :", opt_type ," || Learning Rate: ",learning_rate,"||  Decay : " , decay_rate )
            print('------------------------------------------------------------------------')
            print("Learning Duration : ", round((t_end_g[i] - t_start_g[i]),2),' (secs)',"~ = ",\
                  round((t_end_g[i] - t_start_g[i])/60),"(mins)")
            print('------------------------------------------------------------------------')
            #print("Getting Results  : %0.2f" % round(te_result - ts_result,2),' (sec)')
            print()
            
            
            
            print('========================================================================')
            print("Results:")   
            print('------------------------------------------------------------------------')
            print ('TrainSet Acc = ',round(ac_train_g[i]*100,3),'%')
            print('')
            print('-----------------------') 
            print ('ValidSet Acc = ',round(ac_valid_g[i]*100,3),'%')
            print('')
            print('-----------------------') 
            print('TestSet Acc = ',round(ac_test_g[i]*100,3),'%')
            print('')
            print('-----------------------')
            print('Test-21 Set Acc = ',round(ac_test_21_g[i]*100,3),'%')
            print('')
            #print('-----------------------')
            #print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
            #print('')
            
            print('--------------------->    i=',i , '   <-----------------------')
            print('------------------------------------------------------------------------')
           
#           
#            if (ac_test_21_g[i]*100) > 64:
#                i= repeat_no + 0.5
                
            i+=1
            
            
            #------------------------------------------------------------------
            #+++++++++++++++++++>>       Mean & VAR       <<+++++++++++++++++++
            #------------------------------------------------------------------
            globals()['ac_train_mean_U'+str(num_layers[0])] = np.mean(ac_train_g*100)
            globals()['ac_train_var_U'+str(num_layers[0])] = np.var(ac_train_g*100)
            
            globals()['ac_test_mean_U'+str(num_layers[0])] = np.mean(ac_test_g*100)
            globals()['ac_test_var_U'+str(num_layers[0])] = np.var(ac_test_g*100)
            
            globals()['ac_test_21_mean_U'+str(num_layers[0])] = np.mean(ac_test_21_g*100)
            globals()['ac_test_21_var_U'+str(num_layers[0])] = np.var(ac_test_21_g*100)
            
            #-------------------------------------------------------------------
            #--------------------->>    Print Mean and Var ---------------------
            
            print('---------------->>   Mean And Variance   <<-----------------')
            print('Train Set Mean   = ',\
                  round(globals()['ac_train_mean_U'+str(num_layers[0])] ,3),\
                  ' ||  Train Set Variance   = ', \
                  round(globals()['ac_train_var_U'+str(num_layers[0])],3 ))
            print()
            print('Test Set Mean    = ',\
                  round( globals()['ac_test_mean_U'+str(num_layers[0])] ,3),\
                  ' ||  Test Set Variance    = ', \
                  round(globals()['ac_test_var_U'+str(num_layers[0])],3) )
            print()
            print('Test-21 Set Mean = ', \
                  round(globals()['ac_test_21_mean_U'+str(num_layers[0])],3),\
                  ' ||  Test-21 Set Variance = ',\
                  round(globals()['ac_test_21_var_U'+str(num_layers[0])],3) )
            #______________________________________________________________________________
            #___________________>>   End of GRU Implementation      <<_____________________
            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    



#==============================================================================
#================================ SVC Implementing  ===========================
#==============================================================================
Xtrain_svm = Xtrain.reshape((len(Xtrain),41))
Xvalid_svm = Xvalid.reshape((len(Xvalid),41))
Xtest_svm  = Xtest.reshape((len(Xtest),41))
Xtest_21_svm = Xtest_21.reshape((len(Xtest_21),41))


penalt = 1.0
cache_siz = 10240
coef_0 = 0.0
dec_funct = 'ovr'
deg = 7
gamm = 'auto'
kern = 'sigmoid'
max_iteration = -1
toler = 1e-5
cls_weight =  None

ltime_s = time.time()

clf = SVC(C = penalt ,\
          cache_size = cache_siz, \
          class_weight = cls_weight, \
          coef0 = coef_0,\
          decision_function_shape = dec_funct, \
          degree = deg, \
          gamma = gamm, \
          kernel = kern,\
          max_iter = max_iteration, \
          probability = True, \
          random_state = None, \
          shrinking = False, \
          tol = toler, \
          verbose = True)
clf.fit(Xtrain_svm, Ytrain) 
ltime_e = time.time()
#------------------------------------------------------------------------------
#++++++++++++++++++++++>    Train Set prediction     <+++++++++++++++++++++++++

yh_Xtrain_svm = clf.predict(Xtrain_svm)
#yh_Xtrain_svm = np.around(yh_Xtrain_svm,decimals=0).astype(np.int)
#yh_Xtrain_svm = yh_Xtrain_svm.reshape(len(yh_Xtrain_svm),1)
#
#Ytrain = Ytrain.reshape(len(Ytrain),1)


ac_train_svm = accuracy_score(Ytrain,yh_Xtrain_svm)

#------------------------------------------------------------------------------

#+++++++++++++++++++++>    Test Set prediction     <+++++++++++++++++++++++++++

yh_test_svm = clf.predict(Xtest_svm)
#yh_test_svm = np.around(yh_test_svm,decimals=0).astype(np.int)
#yh_test_svm = yh_test_svm.reshape(len(yh_test_svm),1)
#
#Ytest = Ytest.reshape(len(Ytest),1)


ac_test_svm = accuracy_score(Ytest,yh_test_svm)
#------------------------------------------------------------------------------

#+++++++++++++++++++++>    Test Set prediction     <+++++++++++++++++++++++++++

yh_test_21_svm = clf.predict(Xtest_21_svm)
#yh_test_21_svm = np.around(yh_test_21_svm,decimals=0).astype(np.int)
#yh_test_21_svm = yh_test_21_svm.reshape(len(yh_test_21_svm),1)
#
#Ytest_21 = Ytest_21.reshape(len(Ytest_21),1)


ac_test_21_svm = accuracy_score(Ytest_21,yh_test_21_svm)

#------------------------------------------------------------------------------

#++++++++++++++++++++++>>    Printing Results      <<++++++++++++++++++++++++++
print('========================================================================')
print("Results for Random Forest:")
print('------------------------------------------------------------------------')
print ('TrainSet Acc = ',round(ac_train_svm*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_svm*100,3),'%')
print('')
print('-----------------------')
print('Test-21 Set Acc = ',round(ac_test_21_svm*100,3),'%')
print('')
#_________________________>  End of SVM Implementaion <________________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



#==============================================================================
#================================   Linear SVC  Imp   =========================
#==============================================================================

Xtrain_svm = Xtrain.reshape((len(Xtrain),41))
#Xvalid_svm = Xvalid.reshape((len(Xvalid),41))
Xtest_svm  = Xtest.reshape((len(Xtest),41))
Xtest_21_svm = Xtest_21.reshape((len(Xtest_21),41))

penalt_norm = 'l2'
loss_funct = 'squared_hinge'
dual_opt = False
toler = 1e-5
penalt = 1
multi_cls = 'ovr'
cls_weight = None
max_iteration = -1

ltime_s = time.time()
clf = LinearSVC(C=penalt, \
          class_weight=cls_weight, \
          dual=dual_opt ,\
          fit_intercept=True,\
          intercept_scaling=1,\
          loss=loss_funct, \
          max_iter= max_iteration,\
          multi_class=multi_cls, \
          penalty=penalt_norm, \
          random_state=0, \
          tol=toler, \
          verbose=0)

clf.fit(Xtrain_svm,Ytrain)
ltime_e = time.time()




#==============================================================================
#================================   Random Forest   ===========================
#==============================================================================
Xtrain_svm = Xtrain.reshape((len(Xtrain),41))
#Xvalid_svm = Xvalid.reshape((len(Xvalid),41))
Xtest_svm  = Xtest.reshape((len(Xtest),41))
Xtest_21_svm = Xtest_21.reshape((len(Xtest_21),41))


no_estimators = 240
crtron = 'entropy' # 'gini' #
maximum_depth = None
min_samp_sp= 2
min_samp_lf = 1
min_weight = 0.0
max_feat = None
max_lf_nodes = None
min_impurity_dec = 0.0
min_impurity_sp = 0.0
bootstrp = True
oob_scor = True
no_jobs = -1
random_st = None
verbs = 0
warm_startt = True
class_wght = None



clsf = RandomForestClassifier(n_estimators = no_estimators , \
                              criterion = crtron , \
                              max_depth = maximum_depth , \
                              min_samples_split = min_samp_sp , \
                              min_samples_leaf = min_samp_lf , \
                              min_weight_fraction_leaf = min_weight , \
                              max_features = max_feat , \
                              max_leaf_nodes = max_lf_nodes , \
                              min_impurity_decrease = min_impurity_dec , \
                              min_impurity_split = min_impurity_sp , \
                              bootstrap = bootstrp , \
                              oob_score = oob_scor , \
                              n_jobs = no_jobs , \
                              random_state = random_st , \
                              verbose = verbs , \
                              warm_start = warm_startt , \
                              class_weight = class_wght)

clsf.fit(Xtrain_svm, Ytrain) 

yh_train_dtree = clsf.predict(Xtrain_svm)
ac_train_dtree = accuracy_score(Ytrain,yh_train_dtree)


yh_test_dtree = clsf.predict(Xtest_svm)
ac_test_dtree = accuracy_score(Ytest,yh_test_dtree)

yh_test_21_dtree = clsf.predict(Xtest_21_svm)
ac_test_21_dtree = accuracy_score(Ytest_21,yh_test_21_dtree)




print('========================================================================')
print("Results for Random Forest:")
print('------------------------------------------------------------------------')
print ('TrainSet Acc = ',round(ac_train_dtree*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_dtree*100,3),'%')
print('')
print('-----------------------')
print('Test-21 Set Acc = ',round(ac_test_21_dtree*100,3),'%')
print('')
#print('-----------------------')
#print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#print('')

#print('-------------------->    i = ', i, '    <---------------------')
print('------------------------------------------------------------------------')


        
#______________________>   End of Random Forest   <____________________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#==============================================================================
#=====================>  Combing the Results    <==============================
#==============================================================================


#++++++++++++++++++++>>   OR Logic combing      <<+++++++++++++++++++++++++++++
yh_Xtrain_g = yh_Xtrain_g.reshape(len(yh_Xtrain_g))
yh_test_g = yh_test_g.reshape(len(yh_test_g))
yh_test_21_g = yh_test_21_g.reshape(len(yh_test_21_g))

yh_Xtrain_or = yh_Xtrain_g | yh_Xtrain_c | yh_train_rf
ac_train_or = accuracy_score(Ytrain,yh_Xtrain_or)
ac_train_or_confusion = confusion_matrix (Ytrain,yh_Xtrain_or)

yh_test_or = yh_test_g | yh_test_rf | yh_test_c
ac_test_or = accuracy_score(Ytest,yh_test_or)
ac_test_or_confusion = confusion_matrix (Ytest,yh_test_or)

yh_test_21_or = yh_test_21_c | yh_test_21_g |  yh_test_21_rf
ac_test_21_or = accuracy_score(Ytest_21,yh_test_21_or)
ac_tesr_21_or_confusion = confusion_matrix (Ytest_21,yh_test_21_or)
#______________________________________________________________________________


#++++++++++++++++++++>>   Majority Opinion      <<+++++++++++++++++++++++++++++
yh_Xtrain_major_sum = sum(np.array([yh_Xtrain_c,\
                            yh_Xtrain_g,\
                            yh_train_rf]))
yh_Xtrain_major = np.zeros(len(yh_Xtrain_major_sum))
    
for j in range (len(yh_Xtrain_major_sum)):
    if yh_Xtrain_major_sum[j] > 1 :
        yh_Xtrain_major[j] = 1

ac_train_cmbnd = accuracy_score(Ytrain_1hot,yh_Xtrain_major)

#-----------------------------------------------------------------
yh_test_major_sum = sum(np.array([yh_test_c,\
                            yh_test_g,\
                            yh_test_rf]))
    
yh_test_major = np.zeros(len(yh_test_major_sum))   

for j in range (len(yh_test_major_sum)):
    if yh_test_major_sum[j] > 1:
        yh_test_major[j] = 1      
        
ac_test_cmbnd = accuracy_score(Ytest,yh_test_major)
ac_test_cmbnd_confusion = confusion_matrix (Ytest,yh_test_major)
#------------------------------------------------------------------
yh_test_21_major_sum = sum(np.array([yh_test_21_c,\
                            yh_test_21_g,\
                            yh_test_21_rf]))
    
yh_test_21_major = np.zeros(len(yh_test_21_major_sum))   

for j in range (len(yh_test_21_major_sum)):
    if yh_test_21_major_sum[j] > 1 :
        yh_test_21_major[j] = 1      
        
ac_test_21_cmbnd = accuracy_score(Ytest_21,yh_test_21_major) 
ac_tesr_21_or_confusion = confusion_matrix (Ytest_21,yh_test_21_or)     
#_______________________>   End of getting cmnd results     <__________________
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



#==============================================================================
#====================>>          Printing Results        <<====================
#==============================================================================

print('========================================================================')
print("Results for OR Mode:")
print('----------------------')
print ('TrainSet Acc = ',round(ac_train_or*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_or*100,3),'%')
print('')
print('-----------------------')
print('Test-21 Set Acc = ',round(ac_test_21_or*100,3),'%')
print('')
#print('-----------------------')
#print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#print('')

print('========================================================================')
print("Results for Combined Mode:")
print('-----------------------------')
print ('TrainSet Acc = ',round(ac_train_cmbnd*100,3),'%')
print('')
print('-----------------------')
print('TestSet Acc = ',round(ac_test_cmbnd*100,3),'%')
print('')
print('-----------------------')
print('Test-21 Set Acc = ',round(ac_test_21_cmbnd*100,3),'%')
print('')
#print('-----------------------')
#print('Hard Test Set Acc = ',round(ac_hard_test*100,3),'%')
#print('')

print('------------------------------------------------------------------------')















