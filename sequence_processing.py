import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq


def sentence2char(str_set):
    char_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr)):
            if('N' in sr[i]):
                tmp.append('null')
            else:
                tmp.append(sr[i])
        char_seq.append(' '.join(tmp))
    return char_seq


def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def char2num(charseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(charseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq


def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq


def sentence2num_speid(str_set, tokenizer, MAX_LEN):
    charseq = sentence2char(str_set)
    numseq = char2num(charseq, tokenizer, MAX_LEN)
    return numseq


def get_tokenizer():
    f = ['A', 'C', 'G', 'T']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS,lower=False)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


def get_tokenizer_speid():
    f = ['a', 'c', 'g', 't']
    res = []
    for i in f:
        res.append(i)
    res = np.array(res)
    NB_WORDS = 5
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


def get_data(miRna,lncRna):
    tokenizer = get_tokenizer()
    MAX_LEN = 30
    X_mi = sentence2num(miRna, tokenizer, MAX_LEN)
    MAX_LEN = 4000
    X_lnc = sentence2num(lncRna, tokenizer, MAX_LEN)

    return X_mi, X_lnc


def get_data_speid(enhancers, promoters):
    tokenizer = get_tokenizer_speid()
    MAX_LEN = 3000
    X_en = sentence2num_speid(enhancers, tokenizer, MAX_LEN)
    MAX_LEN = 2000
    X_pr = sentence2num_speid(promoters, tokenizer, MAX_LEN)

    return X_en, X_pr


# In[ ]:



names = ['Arabidopsis lyrata', 'Solanum lycopersicum']
name=names[0]

train_dir='/home/yxy/Project/002/Datasets/Training-validation dataset/'
#imbltrain='/home/hzy/data/%s/imbltrain/'%name
test_dir='/home/yxy/Project/002/Datasets/Test dataset/'
#Data_dir='/home/hzy/data/%s/'%name
Data_dir='/home/yxy/Project/002/processData/'
#print ('Experiment on %s dataset' % name)

print ('Loading seq data...')

miRna_tra = []
lncRna_tra = []
y_tra = []
temp = []
with open(train_dir + 'Sequence(separated with commas).fasta','r') as a:
    for line in a:
        line = line.strip()
        temp.append(line.split(',')[2])
        lncRna_tra.append(line.split(',')[3])
        y_tra.append(int(line.split(',')[4]))
for c in temp:
    miRna_tra.append(c.replace('U','T'))

miRna_tes = []
lncRna_tes = []
y_tes = []
temp = []
with open(test_dir + '%s/Sequence(separated with commas).fasta'%name,'r') as a:
    for line in a:
        line = line.strip()
        temp.append(line.split(',')[2])
        lncRna_tes.append(line.split(',')[3])
        y_tes.append(int(line.split(',')[4]))
for c in temp:
    miRna_tes.append(c.replace('U','T'))

print('平衡训练集')
print('pos_samples:'+ str(sum(y_tra)))
print('neg_samples:'+ str(len(y_tra)-int(sum(y_tra))))
print('测试集')
print('pos_samples:'+ str(int(sum(y_tes))))
print('neg_samples:'+ str(len(y_tes)-int(sum(y_tes))))


X_mi_tra,X_lnc_tra=get_data(miRna_tra,lncRna_tra)
#X_en_imtra,X_pr_imtra=get_data(im_enhancers_tra,im_promoters_tra)
X_mi_tes,X_lnc_tes=get_data(miRna_tes,lncRna_tes)

np.savez(Data_dir+'train624.npz',X_mi_tra=X_mi_tra,X_lnc_tra=X_lnc_tra,y_tra=y_tra)
#np.savez(Data_dir+'im_%s_train.npz'%name,X_en_tra=X_en_imtra,X_pr_tra=X_pr_imtra,y_tra=y_imtra)
np.savez(Data_dir+'%s_test624.npz'%name,X_mi_tes=X_mi_tes,X_lnc_tes=X_lnc_tes,y_tes=y_tes)


