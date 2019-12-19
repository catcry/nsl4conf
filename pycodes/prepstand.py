# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:28:46 2019

@author: catcry
"""


#faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTrain+.csv'
#faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTest+.csv'
#faddr = 'F:\\Git\\nsl4conf\\NSL-KDD\\KDDTest-21.csv'
#return_difficulty = 1


def prep_stand(faddr,seperator,vald,vald_percentage,test,test_percentage,return_difficulty) : 
    import pandas as pd 
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    
    
    data = pd.read_csv(faddr,sep = seperator)
    
    data = data.replace({'protocol_type': 'tcp'}, 0)
    data = data.replace({'protocol_type': 'udp'}, 1)
    data = data.replace({'protocol_type': 'icmp'}, 2)
    
    data = data.replace({'service': 'ftp_data'}, 0)
    data = data.replace({'service': 'other'}, 1)
    data = data.replace({'service': 'private'}, 2)
    data = data.replace({'service': 'http'}, 3)
    data = data.replace({'service': 'remote_job'}, 4)
    data = data.replace({'service': 'name'}, 5)
    data = data.replace({'service': 'netbios_ns'}, 6)
    data = data.replace({'service': 'eco_i'}, 7)
    data = data.replace({'service': 'mtp'}, 8)
    data = data.replace({'service': 'telnet'}, 9)
    data = data.replace({'service': 'finger'}, 10)
    data = data.replace({'service': 'domain_u'}, 11)
    data = data.replace({'service': 'supdup'}, 12)
    data = data.replace({'service': 'uucp_path'}, 13)
    data = data.replace({'service': 'Z39_50'}, 14)
    data = data.replace({'service': 'smtp'}, 15)
    data = data.replace({'service': 'csnet_ns'}, 16)
    data = data.replace({'service': 'uucp'}, 17)
    data = data.replace({'service': 'netbios_dgm'}, 18)
    data = data.replace({'service': 'urp_i'}, 19)
    data = data.replace({'service': 'auth'}, 20)
    data = data.replace({'service': 'domain'}, 21)
    data = data.replace({'service': 'ftp'}, 22)
    data = data.replace({'service': 'bgp'}, 23)
    data = data.replace({'service': 'ldap'}, 24)
    data = data.replace({'service': 'ecr_i'}, 25)
    data = data.replace({'service': 'gopher'}, 26)
    data = data.replace({'service': 'vmnet'}, 27)
    data = data.replace({'service': 'systat'}, 28)
    data = data.replace({'service': 'http_443'}, 29)
    data = data.replace({'service': 'efs'}, 30)
    data = data.replace({'service': 'whois'}, 31)
    data = data.replace({'service': 'imap4'}, 32)
    data = data.replace({'service': 'iso_tsap'}, 33)
    data = data.replace({'service': 'echo'}, 34)
    data = data.replace({'service': 'klogin'}, 35)
    data = data.replace({'service': 'link'}, 36)
    data = data.replace({'service': 'sunrpc'}, 37)
    data = data.replace({'service': 'login'}, 38)
    data = data.replace({'service': 'kshell'}, 39)
    data = data.replace({'service': 'sql_net'}, 40)
    data = data.replace({'service': 'time'}, 41)
    data = data.replace({'service': 'hostnames'}, 42)
    data = data.replace({'service': 'exec'}, 43)
    data = data.replace({'service': 'ntp_u'}, 44)
    data = data.replace({'service': 'discard'}, 45)
    data = data.replace({'service': 'nntp'}, 46)
    data = data.replace({'service': 'courier'},47 )
    data = data.replace({'service': 'ctf'}, 48)
    data = data.replace({'service': 'ssh'}, 49)
    data = data.replace({'service': 'daytime'}, 50)
    data = data.replace({'service': 'shell'}, 51)
    data = data.replace({'service': 'netstat'}, 52)
    data = data.replace({'service': 'pop_3'}, 53)
    data = data.replace({'service': 'nnsp'}, 54)
    data = data.replace({'service': 'IRC'}, 55)
    data = data.replace({'service': 'pop_2'}, 56)
    data = data.replace({'service': 'printer'}, 57)
    data = data.replace({'service': 'tim_i'}, 58)
    data = data.replace({'service': 'pm_dump'}, 59)
    data = data.replace({'service': 'red_i'}, 60)
    data = data.replace({'service': 'netbios_ssn'}, 61)
    data = data.replace({'service': 'rje'}, 62)
    data = data.replace({'service': 'X11'}, 63)
    data = data.replace({'service': 'urh_i'}, 64)
    data = data.replace({'service': 'http_8001'}, 65)
    data = data.replace({'service': 'aol'}, 66)
    data = data.replace({'service': 'http_2784'}, 67)
    data = data.replace({'service': 'tftp_u'}, 68)
    data = data.replace({'service': 'harvest'}, 69)

    data = data.replace({'flag': 'SF'}, 0)
    data = data.replace({'flag': 'S0'}, 1)
    data = data.replace({'flag': 'REJ'}, 2)
    data = data.replace({'flag': 'RSTR'}, 3)
    data = data.replace({'flag': 'SH'}, 4)
    data = data.replace({'flag': 'RSTO'}, 5)
    data = data.replace({'flag': 'S1'}, 6)
    data = data.replace({'flag': 'RSTOS0'}, 7)
    data = data.replace({'flag': 'S3'}, 8)
    data = data.replace({'flag': 'S2'}, 9)
    data = data.replace({'flag': 'OTH'}, 10)
    
    #=====================>       Label Array     <============================
    data = data.replace({'label_train': 'normal'}, 0)
    data = data.replace({'label_train': 'neptune'}, 1)
    data = data.replace({'label_train': 'warezclient'}, 1)
    data = data.replace({'label_train': 'ipsweep'}, 1)
    data = data.replace({'label_train': 'portsweep'}, 1)
    data = data.replace({'label_train': 'teardrop'}, 1)
    data = data.replace({'label_train': 'nmap'}, 1)
    data = data.replace({'label_train': 'satan'}, 1)
    data = data.replace({'label_train': 'smurf'}, 1)
    data = data.replace({'label_train': 'pod'}, 1)
    data = data.replace({'label_train': 'back'}, 1)
    data = data.replace({'label_train': 'guess_passwd'}, 1)
    data = data.replace({'label_train': 'ftp_write'}, 1)
    data = data.replace({'label_train': 'multihop'}, 1)
    data = data.replace({'label_train': 'rootkit'}, 1)
    data = data.replace({'label_train': 'buffer_overflow'}, 1)
    data = data.replace({'label_train': 'imap'}, 1)
    data = data.replace({'label_train': 'warezmaster'}, 1)
    data = data.replace({'label_train': 'phf'}, 1)
    data = data.replace({'label_train': 'land'}, 1)
    data = data.replace({'label_train': 'loadmodule'}, 1)
    data = data.replace({'label_train': 'spy'}, 1)
    data = data.replace({'label_train': 'perl'}, 1)
    data = data.replace({'label_train': 'snmpguess'}, 1)
    data = data.replace({'label_train': 'processtable'}, 1)
    data = data.replace({'label_train': 'back'}, 1)
    data = data.replace({'label_train': 'saint'}, 1)
    data = data.replace({'label_train': 'mscan'}, 1)
    data = data.replace({'label_train': 'apache2'}, 1)
    data = data.replace({'label_train': 'httptunnel'}, 1)
    data = data.replace({'label_train': 'mailbomb'}, 1)
    data = data.replace({'label_train': 'snmpgetattack'}, 1)
    data = data.replace({'label_train': 'worm'}, 1)
    data = data.replace({'label_train': 'sendmail'}, 1)
    data = data.replace({'label_train': 'xlock'}, 1)
    data = data.replace({'label_train': 'xterm'}, 1)
    data = data.replace({'label_train': 'xsnoop'}, 1)
    data = data.replace({'label_train': 'ps'}, 1)
    data = data.replace({'label_train': 'named'}, 1)   
    data = data.replace({'label_train': 'udpstorm'}, 1)
    data = data.replace({'label_train': 'sqlattack'}, 1)
    
    Y = np.array( data['label_train'])
    #Z = np.array(data['difficulty_level'])
    data = data.drop(columns = ['label_train'])
    
    
 #========================>   Normalization   <=============================
    
    scaler = MinMaxScaler()
    
    data[['duration','protocol_type','service','flag','scr_bytes',\
          'dst_bytes','land','wrong_fragment','urgent','hot',\
          'num_failed_logins','logged_in','num_compromised',\
          'root_shell','su_attempted','num_root','num_file_creations',\
          'num_shell','num_access_files','num_outbound_cmds',\
          'is_hot_login','is_guest_login','count','srv_count',\
          'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',\
          'same_srv_rate','diff_srv_rate','srv_diff_host_rate',\
          'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',\
          'dst_host_diff_srv_rate','dst_host_same_src_port_rate',\
          'dst_host_srv_diff_host_rate','dst_host_serror_rate',\
          'dst_host_srv_serror_rate','dst_host_rerror_rate',\
          'dst_host_srv_error_rate']]\
    = scaler.fit_transform(\
    data[['duration','protocol_type','service','flag','scr_bytes',\
          'dst_bytes','land','wrong_fragment','urgent','hot',\
          'num_failed_logins','logged_in','num_compromised',\
          'root_shell','su_attempted','num_root','num_file_creations',\
          'num_shell','num_access_files','num_outbound_cmds',\
          'is_hot_login','is_guest_login','count','srv_count',\
          'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',\
          'same_srv_rate','diff_srv_rate','srv_diff_host_rate',\
          'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',\
          'dst_host_diff_srv_rate','dst_host_same_src_port_rate',\
          'dst_host_srv_diff_host_rate','dst_host_serror_rate',\
          'dst_host_srv_serror_rate','dst_host_rerror_rate',\
          'dst_host_srv_error_rate']]) 
    X = data.to_numpy()
         
    #-------------------------------------------------------------------------- 




    #==========================================================================
    #=========================  Seperatin Vald and Test Sets ==================
    #==========================================================================
    
    if vald == 1 and test == 1 and vald_percentage !=100 and test_percentage != 100:
         X,Xtest,Y,Ytest = train_test_split(X,Y, stratify=Y,test_size=(test_percentage/100))
         Xtrain,Xvalid,Ytrain,Yvalid = train_test_split(X,Y, stratify=Y,\
         test_size=(vald_percentage/100)/(1-test_percentage/100))
         if return_difficulty == 1: 
             Ztrain = Xtrain[:,41]
             Zvalid = Xvalid[:,41]
             Ztest = Xtest [:,41]
         Xtrain = Xtrain[:,0:41]
         Xvalid = Xvalid[:,0:41]
         Xtest = Xtest[:,0:41]
    elif vald == 0 and test == 1 and vald_percentage !=100 and test_percentage != 100:
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y, stratify=Y,test_size=(test_percentage/100))
        Xvalid = 0
        Yvalid = 0
        if return_difficulty == 1: 
             Ztrain = Xtrain[:,41]
             Zvalid = 0
             Ztest = Xtest [:,41]
        Xtrain = Xtrain[:,0:41]
        Xtest = Xtest[:,0:41]
        
    elif vald ==1 and test == 0 and vald_percentage !=100 and test_percentage != 100:
        Xtrain,Xvalid,Ytrain,Yvalid = train_test_split(X,Y, stratify=Y,test_size=(vald_percentage/100))
        Xtest = 0
        Ytest = 0
        if return_difficulty == 1: 
            Ztrain = Xtrain[:,41]
            Zvalid = Xvalid[:,41]
            Ztest = 0
        Xtrain = Xtrain[:,0:41]
        Xvalid = Xvalid[:,0:41]
            
             
    elif vald ==0 and test == 0 and vald_percentage !=100 and test_percentage != 100 :
            Xtrain = X
            Ytrain = Y            
            Xvalid = 0
            Xtest = 0
            Ytest = 0
            Yvalid = 0
            if return_difficulty == 1: 
                Ztrain = Xtrain[:,41]
                Zvalid = 0
                Ztest = 0
            Xtrain = Xtrain[:,0:41]
           
    elif vald ==1 and vald_percentage ==100 or test == 1  and test_percentage == 100 :
            Xtrain = 0
            Ytrain = 0            
            Xvalid = 0
            Yvalid = 0
            Xtest = X
            Ytest = Y
            if return_difficulty == 1: 
                 Ztrain = 0
                 Zvalid = 0
                 Ztest = Xtest [:,41]    
            Xtest = Xtest[:,0:41]
    
    
    if return_difficulty == 1:
        return Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest, Ztrain, Zvalid, Ztest
    
    if return_difficulty == 0: 
        return Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest
                           
    
        
         