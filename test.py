import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from model import get_model
from model import get_model_max
from model import get_model_C_mul
from model import get_model_C_sub
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score, f1_score
from sklearn.metrics import accuracy_score,recall_score


def stat(y_label,y_pred):
    # print('y_label=',y_label)
    # print('y_pred=',y_pred)
    threshold = 0.5
    auc = roc_auc_score(y_label, y_pred)
    aupr = average_precision_score(y_label, y_pred)
    for i in range(len(y_pred)):
        if y_pred[i][0] >= threshold:
            y_pred[i][0] = 1
        if y_pred[i][0] < threshold:
            y_pred[i][0] = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_pred[i][0] == 0 and y_label[i] == 0:
            TN = TN + 1
        if y_pred[i][0] == 1 and y_label[i] == 1:
            TP = TP + 1
        if y_pred[i][0] == 0 and y_label[i] == 1:
            FN = FN + 1
        if y_pred[i][0] == 1 and y_label[i] == 0:
            FP = FP + 1

    
    specificity = TN/(TN+FP)
    recall = recall_score(y_label,y_pred)
    acc = accuracy_score(y_label,y_pred)
    f1 = f1_score(y_label, y_pred)

    acc = round(acc, 4)
    auc = round(auc,4)
    aupr = round(aupr, 4)
    f1 = round(f1,4)

    return acc,auc,aupr,f1,recall,specificity



##########################
datatype = 2021
kmer = 3
##########################



for m in range(100):
    model=None
    model=get_model()
    model.load_weights('./model/3mer2021/Solanum lycopersicumModel%s.h5'%m)
    
    
    if datatype == 2020:
        names = ['Arabidopsis lyrata','Solanum lycopersicum']
    elif datatype == 2021:
        names = ['aly','mtr','stu','bdi']
    
    for name in names:
        Data_dir='/home/yxy/Project/002/processData/3mer/'
        
        if datatype == 2020:
            test=np.load(Data_dir+'5mer%s_test.npz'%name)
        elif datatype == 2021:
            test=np.load(Data_dir+'%s%stest2021.npz'%(name,kmer))
        
        X_mi_tes,X_lnc_tes,y_tes=test['X_mi_tes'],test['X_lnc_tes'],test['y_tes']

        print("****************Testing %s  specific model on %s cell line****************"%(m,name))
        y_pred = model.predict([X_mi_tes,X_lnc_tes])
        auc = roc_auc_score(y_tes, y_pred)
        aupr = average_precision_score(y_tes, y_pred)
        f1 = f1_score(y_tes, np.round(y_pred.reshape(-1)))
        
        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("f1_score", f1)

        acc,auc,aupr,f1,recall,specificity = stat(y_tes, y_pred)
        print("ACC : ", acc,"auc : ", auc,"aupr ：" , aupr,"f1 : ", f1,"recall : ",recall,"specificity : ",specificity)
    

