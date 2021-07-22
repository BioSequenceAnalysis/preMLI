


import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def sentence2word(str_set):
    word_seq=[]
    for sr in str_set:
        tmp=[]
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq

def word2num(wordseq,tokenizer,MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq

def sentence2num(str_set,tokenizer,MAX_LEN):
    wordseq=sentence2word(str_set)
    numseq=word2num(wordseq,tokenizer,MAX_LEN)
    return numseq

def get_tokenizer():
    f= ['A','C','G','T']
    c = itertools.product(f,f,f,f,f,f)
    res=[]
    for i in c:
        temp=i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res=np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS,lower=False)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null']=0
    return tokenizer

def get_data(miRna,incRna):
    tokenizer=get_tokenizer()
    MAX_LEN=30
    X_mi=sentence2num(miRna,tokenizer,MAX_LEN)
    MAX_LEN=4000
    X_inc=sentence2num(incRna,tokenizer,MAX_LEN)

    return X_mi,X_inc


test_dir='/home/yxy/Project/002/Datasets/'
Data_dir='/home/yxy/Project/002/processData/'

print ('Loading seq data...')


new = np.load('TrainingValidationSet2021.npy')
miRna_tra = []
temp = []
lncRna_tra = []
y_tra = []
for sequence in new:
    sequence = sequence.strip()
    temp.append(sequence.split(',')[2])
    lncRna_tra.append(sequence.split(',')[3])
    y_tra.append(int(sequence.split(',')[6]))
for c in temp:
    miRna_tra.append(c.replace('U','T'))

print('训练集')
print('pos_samples:'+ str(int(sum(y_tra))))
print('neg_samples:'+ str(len(y_tra)-int(sum(y_tra))))

X_mi_tra,X_lnc_tra=get_data(miRna_tra,lncRna_tra)
np.savez(Data_dir+'train2021.npz',X_mi_tra=X_mi_tra,X_lnc_tra=X_lnc_tra,y_tra=y_tra)
print("train dataset success")


testnames = ['aly','mtr','stu','bdi']
for name in testnames:
    miRna_tes = []
    lncRna_tes = []
    y_tes = []
    temp = []
    with open(test_dir + '%s-TestSetH.fasta'%name,'r') as a:
        for line in a:
            line = line.strip()
            temp.append(line.split(',')[2])
            lncRna_tes.append(line.split(',')[3])
            y_tes.append(int(line.split(',')[6]))
    for c in temp:
        miRna_tes.append(c.replace('U','T'))

    print('测试集')
    print('pos_samples:'+ str(int(sum(y_tes))))
    print('neg_samples:'+ str(len(y_tes)-int(sum(y_tes))))

    X_mi_tes,X_lnc_tes=get_data(miRna_tes,lncRna_tes)
    np.savez(Data_dir+'%stest2021.npz'%name,X_mi_tes=X_mi_tes,X_lnc_tes=X_lnc_tes,y_tes=y_tes)
    print("%s success"%name)

#print(len(max(miRna_tra, key=len)))
#print(len(max(incRna_tra,key=len)))
#print(len(max(miRna_tes, key=len)))
#print(len(max(incRna_tes,key=len)))



