import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from model import get_model
from model import get_model_max
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.mi = val_data[0]
        self.lnc = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.mi,self.lnc])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        self.model.save_weights(
            "./model/2021bs64/%sModel%d.h5" % (self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
       
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

#names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
# name=names[0]
# The data used here is the sequence processed by data_processing.py.
names = ['Arabidopsis lyrata', 'Solanum lycopersicum']
name=names[1]

Data_dir='/home/yxy/Project/002/processData/'
train = np.load(Data_dir+'train2021.npz')
X_mi_tra, X_lnc_tra, y_tra = train['X_mi_tra'], train['X_lnc_tra'], train['y_tra']

X_mi_tra, X_mi_val,X_lnc_tra,X_lnc_val, y_tra, y_val=train_test_split(
    X_mi_tra,X_lnc_tra,y_tra,test_size=0.1,stratify=y_tra)

model = get_model()
model.summary()
print('Traing %s cell line specific model ...' % name)
back = roc_callback(val_data=[X_mi_val, X_lnc_val, y_val],name=name)
history = model.fit([X_mi_tra, X_lnc_tra], y_tra, validation_data=([X_mi_val, X_lnc_val], y_val), epochs=100, batch_size=32,
                        callbacks=[back])
t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
print("开始时间:"+t1+"结束时间："+t2)
