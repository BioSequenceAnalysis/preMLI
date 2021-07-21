from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras
import numpy as np
import tensorflow as tf

MAX_LEN_mi = 30
MAX_LEN_lnc = 4000

EMBEDDING_DIM = 100
kmers = 6

if kmers == 4:
    NB_WORDS = 257
    lncRnaembedding_matrix = np.load('lnc4mers.npy')
    miRnaembedding_matrix = np.load('mi4mers.npy')
elif kmers == 6:
    NB_WORDS = 4097
    lncRnaembedding_matrix = np.load('lncRnaweight.npy')
    miRnaembedding_matrix = np.load('miRnaweight.npy')
elif kmers == 5:
    NB_WORDS = 1025
    lncRnaembedding_matrix = np.load('./processData/5mer/lnc5mers.npy')
    miRnaembedding_matrix = np.load('./processData/5mer/mi5mers.npy')
elif kmers == 3:
    NB_WORDS = 65
    lncRnaembedding_matrix = np.load('./processData/3mer/lnc3mers.npy')
    miRnaembedding_matrix = np.load('./processData/3mer/mi3mers.npy')

embedding_matrix_one_hot = np.array([[0, 0, 0, 0],
                                     [1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])


class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def metric_F1score(y_true,y_pred):    
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

def metric_recall(y_true,y_pred): 
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall

def get_simCNN():
    enhancer_length = 3000  # TODO: get this from input
    promoter_length = 2000  # TODO: get this from input
    n_kernels = 1024  # Number of kernels; used to be 1024
    filter_length = 40  # Length of each kernel
    LSTM_out_dim = 50  # Output direction of ONE DIRECTION of LSTM; used to be 512
    dense_layer_size = 800
    enhancers = Input(shape=(MAX_LEN_en,))
    promoters = Input(shape=(MAX_LEN_pr,))
    # Convolutional/maxpooling layers to extract prominent motifs
    # Separate identically initialized convolutional layers are trained for
    # enhancers and promoters
    # Define enhancer layers
    enhancer_conv_layer = Convolution1D(input_dim=4,
                                        input_length=enhancer_length,
                                        nb_filter=n_kernels,
                                        filter_length=filter_length,
                                        border_mode="valid",
                                        subsample_length=1,
                                        W_regularizer=l2(1e-5))
    enhancer_max_pool_layer = MaxPooling1D(pool_length=int(
        filter_length / 2), stride=int(filter_length / 2))
    enhancer_length_slim = enhancer_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    enhancer_conv_layer_slim = Convolution1D(input_dim=4,
                                             input_length=enhancer_length_slim,
                                             nb_filter=n_kernels_slim,
                                             filter_length=filter_length_slim,
                                             border_mode="valid",
                                             subsample_length=1,
                                             W_regularizer=l2(1e-5))
    # Build enhancer branch
    enhancer_branch = Sequential()
    enhancer_branch.add(Embedding(5,
                                  4,
                                  weights=[embedding_matrix_one_hot],
                                  input_length=MAX_LEN_en,
                                  trainable=False))
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_conv_layer_slim)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch = enhancer_branch(enhancers)
    # Define promoter layers branch:
    promoter_conv_layer = Convolution1D(input_dim=4,
                                        input_length=promoter_length,
                                        nb_filter=n_kernels,
                                        filter_length=filter_length,
                                        border_mode="valid",
                                        subsample_length=1,
                                        W_regularizer=l2(1e-5))
    promoter_max_pool_layer = MaxPooling1D(pool_length=int(
        filter_length / 2), stride=int(filter_length / 2))
    promoter_length_slim = promoter_length + filter_length - 1
    n_kernels_slim = 200
    filter_length_slim = 20
    promoter_conv_layer_slim = Convolution1D(input_dim=4,
                                             input_length=promoter_length_slim,
                                             nb_filter=n_kernels_slim,
                                             filter_length=filter_length_slim,
                                             border_mode="valid",
                                             subsample_length=1,
                                             W_regularizer=l2(1e-5))
    # Build promoter branch
    promoter_branch = Sequential()
    promoter_branch.add(Embedding(5,
                                  4,
                                  weights=[embedding_matrix_one_hot],
                                  input_length=MAX_LEN_pr,
                                  trainable=False))
    promoter_branch.add(promoter_conv_layer)
    promoter_branch.add(Activation("relu"))
    promoter_branch.add(promoter_conv_layer_slim)
    promoter_branch.add(Activation("relu"))
    promoter_branch.add(promoter_max_pool_layer)
    promoter_branch = promoter_branch(promoters)
    # Define main model layers
    # Concatenate outputs of enhancer and promoter convolutional layers
    #merge_layer = keras.Merge([enhancer_branch, promoter_branch],mode='concat',concat_axis=1)
    merge_all = concatenate([enhancer_branch, promoter_branch], axis=1)
    dense_layer = Dense(output_dim=dense_layer_size,
                        init="glorot_uniform",
                        W_regularizer=l2(1e-6))
    # Logistic regression layer to make final binary prediction
    LR_classifier_layer = Dense(output_dim=1)
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(dense_layer)
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(LR_classifier_layer)
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    pres = model(merge_all)
    model = Model([enhancers, promoters], pres)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(
        lr=1e-5), metrics=['accuracy'])
    return model


def get_speid():
    miRna = Input(shape=(MAX_LEN_mi,))
    lncRna = Input(shape=(MAX_LEN_lnc,))
    enhancer_conv_layer = Convolution1D(input_dim=4,
                                        input_length=3000,
                                        nb_filter=1024,
                                        filter_length=40,
                                        border_mode="valid",
                                        subsample_length=1,
                                        W_regularizer=l2(1e-5))
    enhancer_max_pool_layer = MaxPooling1D(pool_length=20, stride=20)
    enhancer_branch = Sequential()
    enhancer_branch.add(Embedding(5,
                                  4,
                                  weights=[embedding_matrix_one_hot],
                                  input_length=MAX_LEN_en,
                                  trainable=False))
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch = enhancer_branch(enhancers)
    # Build promoter branch
    promoter_conv_layer = Convolution1D(input_dim=4,
                                        input_length=2000,
                                        nb_filter=1024,
                                        filter_length=40,
                                        border_mode="valid",
                                        subsample_length=1,
                                        W_regularizer=l2(1e-5))
    promoter_max_pool_layer = MaxPooling1D(pool_length=20, stride=20)
    # Build promoter branch
    promoter_branch = Sequential()
    promoter_branch.add(Embedding(5,
                                  4,
                                  weights=[embedding_matrix_one_hot],
                                  input_length=MAX_LEN_pr,
                                  trainable=False))
    promoter_branch.add(promoter_conv_layer)
    promoter_branch.add(Activation("relu"))
    promoter_branch.add(promoter_max_pool_layer)
    promoter_branch = promoter_branch(promoters)
    # Define main model layers
    # Concatenate outputs of enhancer and promoter convolutional layers
    merge_layer = concatenate([enhancer_branch, promoter_branch],
                              axis=1)
    biLSTM_layer = Bidirectional(LSTM(input_dim=1024,
                                      output_dim=512,
                                      return_sequences=True))
    dense_layer = Dense(output_dim=800,
                        init="glorot_uniform",
                        W_regularizer=l2(1e-6))
    LR_classifier_layer = Dense(output_dim=1)
   # merge_all = concatenate([conv_enhancers, conv_promoter], axis=2)
    # print(conv_promoter_seq.summary())
    model = Sequential()
    # model.add(merge_layer)
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(biLSTM_layer)
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(dense_layer)
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(LR_classifier_layer)
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    pre = model(merge_layer)
    model = Model([enhancers, promoters], pre)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(
        lr=1e-5), metrics=['accuracy'])
    return model
# print(model.summary())


def get_model():
    miRna = Input(shape=(MAX_LEN_mi,))
    lncRna = Input(shape=(MAX_LEN_lnc,))
    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM,weights=[miRnaembedding_matrix], trainable=True)(miRna)
    #emb_lnc = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[lncRnaembedding_matrix], trainable=True)(lncRna)
    #emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(miRna)
    emb_lnc = Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(lncRna)
    miRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=30,
                                        nb_filter=64,
                                        filter_length=30,
                                        border_mode="same",  # "same"
                                        )
    miRna_max_pool_layer = MaxPooling1D(pool_length=int(1), stride=int(1))

    # Build enhancer branch
    miRna_branch = Sequential()
    miRna_branch.add(miRna_conv_layer)
    miRna_branch.add(Activation("relu"))
    miRna_branch.add(miRna_max_pool_layer)
    miRna_branch.add(BatchNormalization())
    miRna_branch.add(Dropout(0.5))
    miRna_out = miRna_branch(emb_mi)
    lncRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=4000,
                                        nb_filter=64,
                                        filter_length=40,
                                        border_mode="same",

                                        )
    lncRna_max_pool_layer = MaxPooling1D(pool_length=int(20), stride=int(20))

   # promoter_length_slim = 2039
   # n_kernels_slim = 200
   # filter_length_slim = 20
    # Build promoter branch
    lncRna_branch = Sequential()
    lncRna_branch.add(lncRna_conv_layer)
    lncRna_branch.add(Activation("relu"))
    lncRna_branch.add(lncRna_max_pool_layer)
    lncRna_branch.add(BatchNormalization())
    lncRna_branch.add(Dropout(0.5))
    lncRna_out = lncRna_branch(emb_lnc)

    #enhancer_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    #enhancer_max_pool_layer = MaxPooling1D(pool_size = 30, strides = 30)(enhancer_conv_layer)
    #promoter_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    #promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)
    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(miRna_out)
    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(lncRna_out)
    l_att_1 = AttLayer(50)(l_gru_1)
    l_att_2 = AttLayer(50)(l_gru_2)
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])

    #merge_layer=Concatenate(axis=1)([l_att_1, l_att_2, subtract_layer, multiply_layer])
    merge_layer = Concatenate(axis=1)([l_att_1, l_att_2])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    #l_gru = Bidirectional(LSTM(50))(dt)
    #l_att = AttLayer(50)(l_gru)
    #bn2 = BatchNormalization()(l_gru)
    #dt2 = Dropout(0.5)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(output_dim=64, init="glorot_uniform")(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([miRna, lncRna], preds)
    adam = keras.optimizers.adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy',metric_recall,metric_F1score])
    return model


def get_model_C_mul():
    miRna = Input(shape=(MAX_LEN_mi,))
    lncRna = Input(shape=(MAX_LEN_lnc,))
    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM,weights=[
                       miRnaembedding_matrix], trainable=True)(miRna)
    emb_lnc = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       lncRnaembedding_matrix], trainable=True)(lncRna)
    miRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=30,
                                        nb_filter=64,
                                        filter_length=10,
                                        border_mode="same",  # "same"
                                        )
    miRna_max_pool_layer = MaxPooling1D(pool_length=int(1), stride=int(1))

    # Build enhancer branch
    miRna_branch = Sequential()
    miRna_branch.add(miRna_conv_layer)
    miRna_branch.add(Activation("relu"))
    miRna_branch.add(miRna_max_pool_layer)
    miRna_branch.add(BatchNormalization())
    miRna_branch.add(Dropout(0.5))
    miRna_out = miRna_branch(emb_mi)
    lncRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=4000,
                                        nb_filter=64,
                                        filter_length=40,
                                        border_mode="same",

                                        )
    lncRna_max_pool_layer = MaxPooling1D(pool_length=int(20), stride=int(20))


   # promoter_length_slim = 2039
   # n_kernels_slim = 200
   # filter_length_slim = 20
    # Build promoter branch
    lncRna_branch = Sequential()
    lncRna_branch.add(lncRna_conv_layer)
    lncRna_branch.add(Activation("relu"))
    lncRna_branch.add(lncRna_max_pool_layer)
    lncRna_branch.add(BatchNormalization())
    lncRna_branch.add(Dropout(0.5))
    lncRna_out = lncRna_branch(emb_lnc)


    #enhancer_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    #enhancer_max_pool_layer = MaxPooling1D(pool_size = 30, strides = 30)(enhancer_conv_layer)
    #promoter_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    #promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)
    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(miRna_out)
    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(lncRna_out)
    l_att_1 = AttLayer(50)(l_gru_1)
    l_att_2 = AttLayer(50)(l_gru_2)
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])

  
    merge_layer = Concatenate(axis=1)([l_att_1, l_att_2, multiply_layer])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    #l_gru = Bidirectional(LSTM(50))(dt)
    #l_att = AttLayer(50)(l_gru)
    #bn2 = BatchNormalization()(l_gru)
    #dt2 = Dropout(0.5)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(output_dim=64, init="glorot_uniform")(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([miRna, lncRna], preds)
    adam = keras.optimizers.adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy',metric_recall,metric_F1score])
    return model


def get_model_C_sub():
    miRna = Input(shape=(MAX_LEN_mi,))
    lncRna = Input(shape=(MAX_LEN_lnc,))
    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM,weights=[
                       miRnaembedding_matrix], trainable=True)(miRna)
    emb_lnc = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       lncRnaembedding_matrix], trainable=True)(lncRna)
    miRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=30,
                                        nb_filter=64,
                                        filter_length=10,
                                        border_mode="same",  # "same"
                                        )
    miRna_max_pool_layer = MaxPooling1D(pool_length=int(1), stride=int(1))

    # Build enhancer branch
    miRna_branch = Sequential()
    miRna_branch.add(miRna_conv_layer)
    miRna_branch.add(Activation("relu"))
    miRna_branch.add(miRna_max_pool_layer)
    miRna_branch.add(BatchNormalization())
    miRna_branch.add(Dropout(0.5))
    miRna_out = miRna_branch(emb_mi)
    lncRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=4000,
                                        nb_filter=64,
                                        filter_length=40,
                                        border_mode="same",

                                        )
    lncRna_max_pool_layer = MaxPooling1D(pool_length=int(20), stride=int(20))

   # promoter_length_slim = 2039
   # n_kernels_slim = 200
   # filter_length_slim = 20
    # Build promoter branch
    lncRna_branch = Sequential()
    lncRna_branch.add(lncRna_conv_layer)
    lncRna_branch.add(Activation("relu"))
    lncRna_branch.add(lncRna_max_pool_layer)
    lncRna_branch.add(BatchNormalization())
    lncRna_branch.add(Dropout(0.5))
    lncRna_out = lncRna_branch(emb_lnc)

    #enhancer_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    #enhancer_max_pool_layer = MaxPooling1D(pool_size = 30, strides = 30)(enhancer_conv_layer)
    #promoter_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    #promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)
    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(miRna_out)
    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(lncRna_out)
    l_att_1 = AttLayer(50)(l_gru_1)
    l_att_2 = AttLayer(50)(l_gru_2)
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])
    merge_layer = Concatenate(axis=1)([l_att_1, l_att_2, subtract_layer])
    #merge_layer = Concatenate(axis=1)([l_att_1,l_att_2])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    #l_gru = Bidirectional(LSTM(50))(dt)
    #l_att = AttLayer(50)(l_gru)
    #bn2 = BatchNormalization()(l_gru)
    #dt2 = Dropout(0.5)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(output_dim=64, init="glorot_uniform")(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([miRna, lncRna], preds)
    adam = keras.optimizers.adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy',metric_recall,metric_F1score])
    return model


def get_model_max():
    miRna = Input(shape=(MAX_LEN_mi,))
    lncRna = Input(shape=(MAX_LEN_lnc,))
    emb_mi = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       miRnaembedding_matrix], trainable=True)(miRna)
    emb_lnc = Embedding(NB_WORDS, EMBEDDING_DIM, weights=[
                       lncRnaembedding_matrix], trainable=True)(lncRna)
    miRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=30,
                                        nb_filter=64,
                                        filter_length=10,
                                        border_mode="same",  # "same"
                                        )
    miRna_max_pool_layer = MaxPooling1D(pool_length=int(1), stride=int(1))

    # Build enhancer branch
    miRna_branch = Sequential()
    miRna_branch.add(miRna_conv_layer)
    miRna_branch.add(Activation("relu"))
    miRna_branch.add(miRna_max_pool_layer)
    miRna_branch.add(BatchNormalization())
    miRna_branch.add(Dropout(0.5))
    miRna_out = miRna_branch(emb_mi)
    lncRna_conv_layer = Convolution1D(input_dim=100,
                                        input_length=4000,
                                        nb_filter=64,
                                        filter_length=40,
                                        border_mode="same",

                                        )
    lncRna_max_pool_layer = MaxPooling1D(pool_length=int(20), stride=int(20))

   # promoter_length_slim = 2039
   # n_kernels_slim = 200
   # filter_length_slim = 20
    # Build promoter branch
    lncRna_branch = Sequential()
    lncRna_branch.add(lncRna_conv_layer)
    lncRna_branch.add(Activation("relu"))
    lncRna_branch.add(lncRna_max_pool_layer)
    lncRna_branch.add(BatchNormalization())
    lncRna_branch.add(Dropout(0.5))
    lncRna_out = lncRna_branch(emb_lnc)

    #enhancer_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    #enhancer_max_pool_layer = MaxPooling1D(pool_size = 30, strides = 30)(enhancer_conv_layer)
    #promoter_conv_layer = Conv1D(filters = 32,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    #promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)
    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(miRna_out)
    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(lncRna_out)
    l_att_1 = AttLayer(50)(l_gru_1)
    l_att_2 = AttLayer(50)(l_gru_2)
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])

    merge_layer = Concatenate(axis=1)([l_att_1, l_att_2, subtract_layer, multiply_layer])
    #merge_layer = Concatenate(axis=1)([l_att_1,l_att_2])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    #l_gru = Bidirectional(LSTM(50))(dt)
    #l_att = AttLayer(50)(l_gru)
    #bn2 = BatchNormalization()(l_gru)
    #dt2 = Dropout(0.5)(bn2)
    #dt = BatchNormalization()(dt)
    #dt = Dropout(0.5)(dt)
    dt = Dense(output_dim=64, init="glorot_uniform")(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model([miRna, lncRna], preds)
    adam = keras.optimizers.adam(lr=5e-6)
    model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics=['accuracy',metric_recall,metric_F1score])
                
    return model
