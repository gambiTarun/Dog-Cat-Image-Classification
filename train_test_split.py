# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:40:07 2018

@author: ETS-PCBMATE
"""

import os
import shutil as st

def train_test_split(dataloc,test_size,validation=False,validation_size=0):
        
    
    if(validation & bool(validation_size==0)):
            raise ValueError('validation_size to be specified if a validation split required')

    if(bool(not(validation)) & bool(validation_size!=0)):
            raise ValueError('validation_size to be specified if a validation split required only')

    train_size = 1-test_size-validation_size
    
    # getting count in each folder    
    size=[]
    
    for dirs in os.listdir(dataloc):
        if(dirs not in ['train','test','validation']):
            size.append(len(os.listdir(dataloc+'/'+dirs)))
    
    # total size equal to sum check
    sumsize = 0
    
    for i in range(len(size)):
        sumsize += (int(size[i]*(train_size))+int(size[i]*(test_size))+int(size[i]*(validation_size)))
    
    if(sum(size)!=sumsize):
        raise ValueError('Split Sizes provided does not make an integral split between Data items. Please change test or/and validation size.')        
    
    # creating test, train, validation directory
    for dirs in os.listdir(dataloc):
        if(dirs not in ['train','test','validation']):
            os.makedirs(dataloc+'/train/'+dirs,exist_ok=True)
            os.makedirs(dataloc+'/test/'+dirs,exist_ok=True)
            if (validation):
                os.makedirs(dataloc+'/validation/'+dirs,exist_ok=True)
    
    # filling train data       
    i=0      
    for dirs in os.listdir(dataloc):
        c=0
        if(dirs not in ['train','test','validation']):
            for files in os.listdir(dataloc+'/'+dirs):
                c+=1
                st.move(dataloc+'/'+dirs+'/'+files,dataloc+'/'+'train/'+dirs)
                if(c==int(size[i]*(train_size))):
                    i+=1
                    break
                
    # filling test data     
    i=0        
    for dirs in os.listdir(dataloc):
        c=0
        if(dirs not in ['train','test','validation']):
            for files in os.listdir(dataloc+'/'+dirs):
                c+=1
                st.move(dataloc+'/'+dirs+'/'+files,dataloc+'/'+'test/'+dirs)
                if(c==int(size[i]*(test_size))):
                    i+=1
                    break

    # filling validation data
    if(validation):
        i=0
        for dirs in os.listdir(dataloc):
            c=0
            if(dirs not in ['train','test','validation']):
                for files in os.listdir(dataloc+'/'+dirs):
                    c+=1
                    st.move(dataloc+'/'+dirs+'/'+files,dataloc+'/'+'validation/'+dirs)
                    if(c==int(size[i]*(validation_size))):
                        i+=1
                        break
    
    # deleting previous empty directories
    for dirs in os.listdir(dataloc):
        if not os.listdir(dataloc+'/'+dirs):
            os.rmdir(dataloc+'/'+dirs)

def main():
    train_test_split('run2',test_size=0.2,validation=True,validation_size=.1)
    
if __name__ == '__main__':
    main()