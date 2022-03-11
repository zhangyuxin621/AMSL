import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.losses import categorical_crossentropy
from keras.backend import permute_dimensions
from keras.activations import softmax, relu
from keras.initializers import RandomUniform
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
import keras.backend.tensorflow_backend as KTF
import numpy as np
import scipy.io as scio
import random
import math
import os
from sklearn import metrics
import argparse
from pyimagesearch.learning_rate_schedulers import StepDecay
from pyimagesearch.learning_rate_schedulers import PolynomialDecay
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100,help="# of epochs to train for")
ap.add_argument("-e", "--transformation", type=int, default=7,help="# of transformation number")
ap.add_argument("-e", "--input_dim_x", type=int, default=125,help="# of input_dim_x")
ap.add_argument("-e", "--input_dim_y", type=int, default=45,help="# of input_dim_y")
ap.add_argument("-e", "--global_mem_dim", type=int, default=500,help="# of global_mem_dim")
ap.add_argument("-e", "--local_mem_dim", type=int, default=500,help="# of local_mem_dim")
ap.add_argument("-e", "--filter_size0", type=int, default=1,help="# of filter_size0")
ap.add_argument("-e", "--filter_size1", type=int, default=32,help="# of filter_size1")
ap.add_argument("-e", "--filter_size2", type=int, default=64,help="# of filter_size2")
ap.add_argument("-e", "--filter_size3", type=int, default=128,help="# of filter_size3")
ap.add_argument("-e", "--lambda1", type=float, default=1.0,help="# of lambda1")
ap.add_argument("-e", "--lambda2", type=float, default=0.0002,help="# of lambda2")
ap.add_argument("-e", "--batch", type=int, default=16,help="# of batch size")
ap.add_argument('--data_path', type = str, default = '/media/zyx/self_supervised/DSADS/dataset_normalize_together/',
                   help='path to load data')
ap.add_argument('--model_path', type = str, default = '/media/zyx/self_supervised/DSADS/model_train/',
                   help='path to save model')

args = vars(ap.parse_args())
epochs = args["epochs"]
trans = args["transformation"]
input_dim_x = args["input_dim_x"]
input_dim_y = args["input_dim_y"]
global_mem_dim = args["global_mem_dim"]
local_mem_dim = args["local_mem_dim"]
filter_size0 = args["filter_size0"]
filter_size1 = args["filter_size1"]
filter_size2 = args["filter_size2"]
filter_size3 = args["filter_size3"]
l1 = args["lambda1"]
l2 = args["lambda2"]
batch = args["batch"]
path = args["data_path"]
model_path = args["model_path"]

class Memory_global(Layer):
    def __init__(self, mem_dim, fea_dim,**kwargs):
        # C: dimension of vector z
        # M: size of the memory
        super(Memory_global, self).__init__(**kwargs)
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.kernel_regularizer = None
        self.std = 1. / math.sqrt(self.fea_dim)

    def build(self, input_shape):
        # M x C
        self.weight = self.add_weight(name='kernel',shape=(self.fea_dim,self.mem_dim),
                                      initializer=RandomUniform(-self.std, self.std),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(Memory_global, self).build(input_shape)

    def get_config(self):
        config = {'mem_dim': self.mem_dim,'fea_dim': self.fea_dim}
        base_config = super(Memory_global, self).get_config()
        print(base_config)
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x1_size = K.int_shape(inputs)[1]
        x2_size = K.int_shape(inputs)[2]
        channel = K.int_shape(inputs)[3]

        inputs = K.reshape(inputs,[-1,channel])

        distance = K.dot(inputs, K.transpose(self.weight))  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = softmax(distance,axis=1)

        output = K.dot(att_weight, self.weight)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output =K.reshape(output,[-1, x1_size, x2_size, channel])
        att = tf.reshape(att_weight, [-1,  x1_size* x2_size* self.fea_dim])
        att =K.mean(-att * K.log(att), axis=-1, keepdims=True)
        return [output, att]

    def compute_output_shape(self, input_shape):
       #assert isinstance(input_shape, list)
        #return [(input_shape[0],input_shape[1],input_shape[2],input_shape[3]),(input_shape[0],input_shape[1],input_shape[2],self.mem_dim)]
        return [(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
                (input_shape[0],1)]

class Memory_local(Layer):
    def __init__(self, mem_dim, fea_dim,**kwargs):
        # C: dimension of vector z
        # M: size of the memory
        super(Memory_local, self).__init__(**kwargs)
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.kernel_regularizer = None
        self.std = 1. / math.sqrt(self.fea_dim)

    def build(self, input_shape):
        # M x C
        self.weight = self.add_weight(name='kernel_local',shape=(self.fea_dim,self.mem_dim),
                                      initializer=RandomUniform(-self.std, self.std),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(Memory_local, self).build(input_shape)

    def get_config(self):
        config = {'mem_dim': self.mem_dim,'fea_dim': self.fea_dim}
        base_config = super(Memory_local, self).get_config()
        print(base_config)
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x1_size = K.int_shape(inputs)[1]
        x2_size = K.int_shape(inputs)[2]
        channel = K.int_shape(inputs)[3]

        inputs = K.reshape(inputs,[-1,channel])

        distance = K.dot(inputs, K.transpose(self.weight))  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = softmax(distance,axis=1)

        output = K.dot(att_weight, self.weight)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output =K.reshape(output,[-1, x1_size, x2_size, channel])
        att = tf.reshape(att_weight, [-1,  x1_size* x2_size* self.fea_dim])
        att = K.mean(-att * K.log(att), axis=-1, keepdims=True)
        return [output, att]

    def compute_output_shape(self, input_shape):
       #assert isinstance(input_shape, list)
        #return [(input_shape[0],input_shape[1],input_shape[2],input_shape[3]),(input_shape[0],input_shape[1],input_shape[2],self.mem_dim)]
        return [(input_shape[0], input_shape[1], input_shape[2], input_shape[3]),
                (input_shape[0],1)]

def slice(x, index):
    return x[:, index, :, :, :]

def mse_compute(x,index):
    encoder = x[0][:,index,:,:,:]
    decoder = x[1]
    return K.mean(K.square(encoder-decoder))

def conlstm_auto():
    print("training start")
    #############################encoder########################
    encoder_input1 = Input(shape=(trans, input_dim_x,input_dim_y, 1),name="encoder_input_no")
    x1 = TimeDistributed(ZeroPadding2D(padding =((3,0),(3,0)) , data_format="channels_last", name='zero'), name='T0')(encoder_input1)
    x1 =TimeDistributed(Conv2D(filter_size1, (4, 4), activation='relu', padding='same', data_format="channels_last",name = 'conv1'),name='T1')(x1)
    x1 = TimeDistributed(MaxPooling2D((2,2), padding='same', name='pool1'), name='T2')(x1)
    x1 = TimeDistributed(Conv2D(filter_size2, (4, 4), activation='relu', padding='same', data_format="channels_last",name = 'conv2'),name='T3')(x1)
    encoded1 = TimeDistributed(MaxPooling2D((2,2), padding='same', name='pool2'), name='T4')(x1)

    x1_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 0},name='L1')(encoded1)
    x2_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 1},name='L2')(encoded1)
    x3_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 2},name='L3')(encoded1)
    x4_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 3},name='L4')(encoded1)
    x5_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 4},name='L5')(encoded1)
    x6_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 5},name='L6')(encoded1)
    x7_ = Lambda(slice, output_shape=(32, 12, filter_size2), arguments={'index': 6},name='L7')(encoded1)

    #############################self supervision########################
    inp1 = Input(shape = (32, 12, filter_size2), name='global_class')
    predict = Conv2D(1, (4, 4), padding='same', activation='sigmoid', data_format="channels_last")(inp1)
    predict = Flatten()(predict)
    predict = Dense(128, activation='relu')(predict)
    predict = Dropout(0.5)(predict)
    predict = Dense(7, activation='softmax')(predict)
    model_class = Model(inputs=inp1, outputs=predict,name = 'model_class')

    g1 = model_class(x1_)
    g2 = model_class(x2_)
    g3 = model_class(x3_)
    g4 = model_class(x4_)
    g5 = model_class(x5_)
    g6 = model_class(x6_)
    g7 = model_class(x7_)

    #############################global memory########################
    inp = Input(shape = (32, 12, filter_size2), name='global_input')
    memory_output, att_weight = Memory_local(mem_dim=filter_size2, fea_dim=global_mem_dim)(inp)
    model_global = Model(inputs=inp, outputs=[memory_output,att_weight],name='global_memory')

    memory_output_g1, att_weight_g1 = model_global(x1_)
    memory_output_g2, att_weight_g2 = model_global(x2_)
    memory_output_g3, att_weight_g3 = model_global(x3_)
    memory_output_g4, att_weight_g4 = model_global(x4_)
    memory_output_g5, att_weight_g5 = model_global(x5_)
    memory_output_g6, att_weight_g6 = model_global(x6_)
    memory_output_g7, att_weight_g7 = model_global(x7_)

    memory_global_sparse = Add()(
        [att_weight_g1, att_weight_g2, att_weight_g3, att_weight_g4, att_weight_g5, att_weight_g6, att_weight_g7])

    #############################local memory########################
    memory_output1, att_weight1 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_1')(x1_)
    memory_output2, att_weight2 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_2')(x2_)
    memory_output3, att_weight3 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_3')(x3_)
    memory_output4, att_weight4 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_4')(x4_)
    memory_output5, att_weight5 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_5')(x5_)
    memory_output6, att_weight6 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_6')(x6_)
    memory_output7, att_weight7 = Memory_local(mem_dim=filter_size2, fea_dim=local_mem_dim,name='memory_local_7')(x7_)

    memory_local_sparse = Add()(
        [att_weight1, att_weight2, att_weight3, att_weight4, att_weight5, att_weight6, att_weight7])

    #############################adaptive fusion########################
    c = Input(shape=(1,))
    c1 = Dense(2*trans)(c)
    c1 = BatchNormalization(momentum=0.93)(c1,training=False)
    c1 = Activation('sigmoid')(c1)

    m_out1 = Lambda(lambda x: x[:,0], name='c1')(c1)
    m_out2 = Lambda(lambda x: x[:,1], name='c2')(c1)
    local_weight1 = Multiply(name='weight_l1')([memory_output1, m_out1])
    global_weight1 = Multiply(name='weight_g1')([memory_output_g1, m_out2])
    final_1 = Add()([local_weight1,global_weight1])

    m_out3 = Lambda(lambda x: x[:,2], name='c3')(c1)
    m_out4 = Lambda(lambda x: x[:,3], name='c4')(c1)
    local_weight2 = Multiply(name='weight_l2')([memory_output2, m_out3])
    global_weight2 = Multiply(name='weight_g2')([memory_output_g2, m_out4])
    final_2 = Add()([local_weight2,global_weight2])

    m_out5 = Lambda(lambda x: x[:,4], name='c5')(c1)
    m_out6 = Lambda(lambda x: x[:,5], name='c6')(c1)
    local_weight3 = Multiply(name='weight_l3')([memory_output3, m_out5])
    global_weight3 = Multiply(name='weight_g3')([memory_output_g3, m_out6])
    final_3 = Add()([local_weight3,global_weight3])

    m_out7 = Lambda(lambda x: x[:,6], name='c7')(c1)
    m_out8 = Lambda(lambda x: x[:,7], name='c8')(c1)
    local_weight4 = Multiply(name='weight_l4')([memory_output4, m_out7])
    global_weight4 = Multiply(name='weight_g4')([memory_output_g4, m_out8])
    final_4 = Add()([local_weight4,global_weight4])

    m_out9 = Lambda(lambda x: x[:,8], name='c9')(c1)
    m_out10 = Lambda(lambda x: x[:,9], name='c10')(c1)
    local_weight5 = Multiply(name='weight_l5')([memory_output5, m_out9])
    global_weight5 = Multiply(name='weight_g5')([memory_output_g5, m_out10])
    final_5 = Add()([local_weight5,global_weight5])

    m_out11 = Lambda(lambda x: x[:,10], name='c11')(c1)
    m_out12 = Lambda(lambda x: x[:,11], name='c12')(c1)
    local_weight6 = Multiply(name='weight_l6')([memory_output1, m_out11])
    global_weight6 = Multiply(name='weight_g6')([memory_output_g1, m_out12])
    final_6 = Add()([local_weight6,global_weight6])

    m_out13 = Lambda(lambda x: x[:,12], name='c13')(c1)
    m_out14 = Lambda(lambda x: x[:,13], name='c14')(c1)
    local_weight7 = Multiply(name='weight_l7')([memory_output1, m_out13])
    global_weight7 = Multiply(name='weight_g7')([memory_output_g1, m_out14])
    final_7 = Add()([local_weight7,global_weight7])

    memory_output_raw = Concatenate(axis=-1,name='concat_local1_uni')([x1_, final_1])
    memory_output_no = Concatenate(axis=-1,name='concat_local2_uni')([x2_, final_2])
    memory_output_ne = Concatenate(axis=-1,name='concat_local3_uni')([x3_, final_3])
    memory_output_op = Concatenate(axis=-1,name='concat_local4_uni')([x4_, final_4])
    memory_output_pe = Concatenate(axis=-1,name='concat_local5_uni')([x5_, final_5])
    memory_output_sc = Concatenate(axis=-1,name='concat_local6_uni')([x6_, final_6])
    memory_output_ti = Concatenate(axis=-1,name='concat_local7_uni')([x7_, final_7])

    #############################decoder########################

    xx1_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose1_1')(memory_output_raw)
    xx1_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose1_2')(xx1_l)
    xx1_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose1_3')(xx1_l)
    xx1_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose1_4')(xx1_l)
    decoder1_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose1_5')(xx1_l)

    xx2_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose2_1')(memory_output_no)
    xx2_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose2_2')(xx2_l)
    xx2_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose2_3')(xx2_l)
    xx2_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose2_4')(xx2_l)
    decoder2_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose2_5')(xx2_l)

    xx3_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose3_1')(memory_output_ne)
    xx3_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose3_2')(xx3_l)
    xx3_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose3_3')(xx3_l)
    xx3_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose3_4')(xx3_l)
    decoder3_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose3_5')(xx3_l)

    xx4_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose4_1')(memory_output_op)
    xx4_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose4_2')(xx4_l)
    xx4_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose4_3')(xx4_l)
    xx4_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose4_4')(xx4_l)
    decoder4_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose4_5')(xx4_l)

    xx5_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose5_1')(memory_output_pe)
    xx5_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose5_2')(xx5_l)
    xx5_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose5_3')(xx5_l)
    xx5_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose5_4')(xx5_l)
    decoder5_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose5_5')(xx5_l)

    xx6_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose6_1')(memory_output_sc)
    xx6_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose6_2')(xx6_l)
    xx6_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose6_3')(xx6_l)
    xx6_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose6_4')(xx6_l)
    decoder6_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose6_5')(xx6_l)

    xx7_l = Conv2DTranspose(filter_size3, (4, 4), padding='same', activation='relu',name='transpose7_1')(memory_output_ti)
    xx7_l = Conv2DTranspose(filter_size2, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose7_2')(xx7_l)
    xx7_l = Conv2DTranspose(filter_size1, (4, 4), padding='same', strides=(2, 2), activation='relu',name='transpose7_3')(xx7_l)
    xx7_l = Conv2DTranspose(filter_size0, (4, 4), padding='same', activation='sigmoid',name='transpose7_4')(xx7_l)
    decoder7_l = Cropping2D(cropping=((3, 0), (3, 0)), data_format="channels_last",name='transpose7_5')(xx7_l)

    mse_loss1_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 0},name='mse1')([encoder_input1, decoder1_l])
    mse_loss2_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 1},name='mse2')([encoder_input1, decoder2_l])
    mse_loss3_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 2},name='mse3')([encoder_input1, decoder3_l])
    mse_loss4_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 3},name='mse4')([encoder_input1, decoder4_l])
    mse_loss5_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 4},name='mse5')([encoder_input1, decoder5_l])
    mse_loss6_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 5},name='mse6')([encoder_input1, decoder6_l])
    mse_loss7_l = Lambda(mse_compute, output_shape=(1,), arguments={'index': 6},name='mse7')([encoder_input1, decoder7_l])

    mse_loss = Add()([mse_loss1_l, mse_loss2_l, mse_loss3_l, mse_loss4_l, mse_loss5_l, mse_loss6_l, mse_loss7_l])

    sparse_loss = Add()([memory_global_sparse, memory_local_sparse])

    COMPOSITE_ED = Model(inputs=[encoder_input1,c],
                         outputs=[mse_loss,sparse_loss,g1,g2,g3,g4,g5,g6,g7])

    COMPOSITE_ED.compile(loss=[lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred,'categorical_crossentropy',
                               'categorical_crossentropy','categorical_crossentropy','categorical_crossentropy',
                               'categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],
                         loss_weights=[1, l2, l1, l1, l1, l1, l1, l1, l1], optimizer='Adam')
    return COMPOSITE_ED

if __name__ == '__main__':
    model = conlstm_auto()

    X_train_raw = np.load(path + "data_raw_train.npy")
    X_train_no = np.load(path + "data_no_train.npy")
    X_train_ne = np.load(path + "data_ne_train.npy")
    X_train_op = np.load(path + "data_op_train.npy")
    X_train_pe = np.load(path + "data_pe_train.npy")
    X_train_sc = np.load(path + "data_sc_train.npy")
    X_train_ti = np.load(path + "data_ti_train.npy")

    X_train = np.concatenate((X_train_raw,X_train_no,X_train_ne,X_train_op,X_train_pe,X_train_sc,X_train_ti),axis=-1)
    X_train = X_train.transpose(0,-1,1,2)
    X_train = np.reshape(X_train, X_train.shape + (1,))

    #############################model train#########################

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    filepath = model_path+'model_best_weight.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True,
                                 save_best_only=True, mode='min')

    dataY1 = np.zeros((X_train.shape[0], 1))
    initial_c = np.zeros((X_train.shape[0], 1))

    label = []
    for i in range(trans):
        for j in range(X_train.shape[0]):
            label.append(i)
    label = np.array(label)
    y_classes = np_utils.to_categorical(label)
    n = X_train.shape[0]
    y_1 = y_classes[:n]
    y_2= y_classes[n:n*2]
    y_3= y_classes[n*2:n*3]
    y_4= y_classes[n*3:n*4]
    y_5= y_classes[n*4:n*5]
    y_6= y_classes[n*5:n*6]
    y_7= y_classes[n*6:n*7]

    history = model.fit([X_train,initial_c],
                        [dataY1, dataY1,y_1,y_2,y_3,y_4,y_5,y_6,y_7], epochs=epochs, batch_size=batch, callbacks=checkpoint, validation_split=0.2)

    [predict_label8, predict_label9, predict_label1, predict_label2,predict_label3,predict_label4,predict_label5, predict_label6, predict_label7] = model.predict([X_train,initial_c], batch_size=batch, verbose=1)

    class_loss = predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] + predict_label4[:,
                                                                                      3] + predict_label5[:,
                                                                                           4] + predict_label6[:,
                                                                                                5] + predict_label7[:,
                                                                                                     6]
    np.savetxt(model_path + 'train_normal_loss_sparse.csv', predict_label9,delimiter=',')
    np.savetxt(model_path + 'train_normal_loss_class.csv', class_loss, delimiter=',')
    np.savetxt(model_path + 'train_normal_loss_sum_mse.csv', predict_label8, delimiter=',')

    #############################model test#########################
    X_test_raw = np.load(path + "data_raw_test.npy")
    X_test_no = np.load(path + "data_no_test.npy")
    X_test_ne = np.load(path + "data_ne_test.npy")
    X_test_op = np.load(path + "data_op_test.npy")
    X_test_pe = np.load(path + "data_pe_test.npy")
    X_test_sc = np.load(path + "data_sc_test.npy")
    X_test_ti = np.load(path + "data_ti_test.npy")

    X_test = np.concatenate((X_test_raw, X_test_no,X_test_ne,X_test_op,X_test_pe,X_test_sc,X_test_ti),axis=-1)
    X_test = X_test.transpose(0,-1,1,2)
    X_test = np.reshape(X_test, X_test.shape + (1,))
    initial_c_test = np.zeros((X_test.shape[0], 1))

    [predict_label8,predict_label9, predict_label1, predict_label2,predict_label3,predict_label4,predict_label5, predict_label6, predict_label7] = model.predict([X_test,initial_c_test], batch_size=batch, verbose=1)

    class_loss = predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] + predict_label4[:,
                                                                                      3] + predict_label5[:,
                                                                                           4] + predict_label6[:,
                                                                                                5] + predict_label7[:,
                                                                                                     6]

    np.savetxt(model_path+ 'normal_loss_sparse.csv', predict_label9, delimiter=',')
    np.savetxt(model_path+'normal_loss_class.csv', class_loss, delimiter=',')
    np.savetxt(model_path+ 'normal_loss_sum_mse.csv', predict_label8, delimiter=',')

    abnormal_s_raw = np.load(path + "data_raw_abnormal.npy")
    abnormal_s_no = np.load(path + "data_no_abnormal.npy")
    abnormal_s_ne = np.load(path + "data_ne_abnormal.npy")
    abnormal_s_op = np.load(path + "data_op_abnormal.npy")
    abnormal_s_pe = np.load(path + "data_pe_abnormal.npy")
    abnormal_s_sc = np.load(path + "data_sc_abnormal.npy")
    abnormal_s_ti = np.load(path + "data_ti_abnormal.npy")

    abnormal = np.concatenate((abnormal_s_raw, abnormal_s_no,abnormal_s_ne,abnormal_s_op,abnormal_s_pe,abnormal_s_sc,abnormal_s_ti), axis=-1)
    abnormal = abnormal.transpose(0, -1, 1, 2)
    abnormal = np.reshape(abnormal, abnormal.shape + (1,))
    initial_c_ab = np.zeros((abnormal.shape[0], 1))

    [predict_label8, predict_label9,predict_label1, predict_label2,predict_label3,predict_label4,predict_label5, predict_label6, predict_label7] = model.predict([abnormal,initial_c_ab], batch_size=batch,verbose=1)

    class_loss = predict_label1[:, 0] + predict_label2[:, 1] + predict_label3[:, 2] + predict_label4[:,
                                                                                      3] + predict_label5[:,
                                                                                           4] + predict_label6[:,
                                                                                                5] + predict_label7[:,
                                                                                                     6]

    np.savetxt(model_path + 'abnormal_loss_sparse.csv', predict_label9,delimiter=',')
    np.savetxt(model_path + 'abnormal_loss_class.csv', class_loss, delimiter=',')
    np.savetxt(model_path + 'abnormal_loss_sum_mse.csv', predict_label8, delimiter=',')

