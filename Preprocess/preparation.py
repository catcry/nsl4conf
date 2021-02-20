# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:05:06 2019

@author: catcry
"""

from nsl4conf.Preprocess.preprocess import prep_stand, prep_manual
faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTrain+.csv'
seperator =  ','

#==============================================================================
#========================>  Regular Data Preparation   <=======================
#==============================================================================


[Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest,Ztrain,Zvalid,Ztest] = \
                                prep_stand (faddr,seperator,1,10,0,0,1)

faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTest+.csv'
faddr = '/home/catcry/Downloads/NSL-KDD/KDDTest+.txt'

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
