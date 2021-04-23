import numpy as np
import os
import scipy.io as scio
import random
from random import choice
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing

global a,dim_DA
a = 125
dim_DA = 45

def inter_data(hr, window=11):
    N = window
    time3 = savgol_filter(hr, window_length=N, polyorder=2)
    return time3

def noised(signal):
    SNR = 5
    noise = np.random.randn(signal.shape[0], signal.shape[1])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise

def negated(signal):
    return signal * -1

def opposite_time(signal):
    return signal[::-1,:]

def permuted(signal):
    listA = [0,1,2,3,4]
    random.shuffle(listA)
    sig = signal[listA[0]*25:listA[0]*25+25]
    for i in range(1,len(listA)):
        sig = np.vstack((sig,signal[listA[i]*25:listA[i]*25+25]))
    return sig

def scale(signal):
    sc = [0.5, 2, 1.5, 0.8]
    s = choice(sc)
    return signal * s

def time_warp(signal):
    for i in range(signal.shape[1]):
        signal[:,i] = inter_data(signal[:,i],11)
    return signal

def regular_mm(data):
    dim = dim_DA * a
    data = data.reshape(data.shape[0], dim)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = data.reshape(data.shape[0], a, dim_DA)
    return data

def transformation(dataX):
    data_no = np.zeros((dataX.shape[0],dataX.shape[1],dataX.shape[2]))
    data_ne = np.zeros((dataX.shape[0],dataX.shape[1],dataX.shape[2]))
    data_op = np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2]))
    data_pe = np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2]))
    data_sc = np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2]))
    data_ti = np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2]))

    for i in range(dataX.shape[0]):
        data_no[i] = noised(dataX[i].copy())
        data_ne[i] = negated(dataX[i].copy())
        data_op[i] = opposite_time(dataX[i].copy())
        data_pe[i] = permuted(dataX[i].copy())
        data_sc[i] = scale(dataX[i].copy())
        data_ti[i] = time_warp(dataX[i].copy())

    #####################Normalization###########################
    data_raw = regular_mm(dataX)
    data_no = regular_mm(data_no)
    data_ne = regular_mm(data_ne)
    data_op= regular_mm(data_op)
    data_pe = regular_mm(data_pe)
    data_sc = regular_mm(data_sc)
    data_ti = regular_mm(data_ti)

    data_raw = np.reshape(data_raw, data_raw.shape + (1,))
    data_no = np.reshape(data_no, data_no.shape + (1,))
    data_ne = np.reshape(data_ne, data_ne.shape + (1,))
    data_op = np.reshape(data_op, data_op.shape + (1,))
    data_pe = np.reshape(data_pe, data_pe.shape + (1,))
    data_sc = np.reshape(data_sc, data_sc.shape + (1,))
    data_ti = np.reshape(data_ti, data_ti.shape + (1,))

    return data_raw,data_no,data_ne,data_op,data_pe,data_sc,data_ti

def shuffle(normal_s_raw,normal_s_no,normal_s_ne,normal_s_op,normal_s_pe,normal_s_sc,normal_s_ti):
    path = '/media/zyx/self_supervised/DSADS/dataset_normalize_together/'
    ######################shuffle data#########################
    listA = [l for l in range(normal_s_no.shape[0])]
    random.shuffle(listA)
    listB = [p for p in range(normal_s_no.shape[0])]

    dataset_raw = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_no = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_ne = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_op = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_pe = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_sc = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])
    dataset_ti = np.zeros(
        [normal_s_no.shape[0], normal_s_no.shape[1], normal_s_no.shape[2], normal_s_no.shape[3]])

    for w, r in zip(listA, listB):
        dataset_raw[r, :, :, :] = normal_s_raw[w, :, :, :]
        dataset_no[r, :, :, :] = normal_s_no[w, :, :, :]
        dataset_ne[r, :, :, :] = normal_s_ne[w, :, :, :]
        dataset_op[r, :, :, :] = normal_s_op[w, :, :, :]
        dataset_pe[r, :, :, :] = normal_s_pe[w, :, :, :]
        dataset_sc[r, :, :, :] = normal_s_sc[w, :, :, :]
        dataset_ti[r, :, :, :] = normal_s_ti[w, :, :, :]

    print('shuffle done')

    X_train_raw = dataset_raw[:int(dataset_raw.shape[0] * 0.6), :, :, :]
    X_test_raw = dataset_raw[int(dataset_raw.shape[0] * 0.6):, :, :, :]

    X_train_no = dataset_no[:int(dataset_no.shape[0] * 0.6), :, :, :]
    X_test_no = dataset_no[int(dataset_no.shape[0] * 0.6):, :, :, :]

    X_train_ne = dataset_ne[:int(dataset_ne.shape[0] * 0.6), :, :, :]
    X_test_ne = dataset_ne[int(dataset_ne.shape[0] * 0.6):, :, :, :]

    X_train_op = dataset_op[:int(dataset_op.shape[0] * 0.6), :, :, :]
    X_test_op = dataset_op[int(dataset_op.shape[0] * 0.6):, :, :, :]

    X_train_pe = dataset_pe[:int(dataset_pe.shape[0] * 0.6), :, :, :]
    X_test_pe = dataset_pe[int(dataset_pe.shape[0] * 0.6):, :, :, :]

    X_train_sc = dataset_sc[:int(dataset_sc.shape[0] * 0.6), :, :, :]
    X_test_sc = dataset_sc[int(dataset_sc.shape[0] * 0.6):, :, :, :]

    X_train_ti = dataset_ti[:int(dataset_ti.shape[0] * 0.6), :, :, :]
    X_test_ti = dataset_ti[int(dataset_ti.shape[0] * 0.6):, :, :, :]

    print('save preparing')

    np.save(path + "data_raw_train.npy", X_train_raw)
    np.save(path + "data_no_train.npy", X_train_no)
    np.save(path + "data_ne_train.npy", X_train_ne)
    np.save(path + "data_op_train.npy", X_train_op)
    np.save(path + "data_pe_train.npy", X_train_pe)
    np.save(path + "data_sc_train.npy", X_train_sc)
    np.save(path + "data_ti_train.npy", X_train_ti)

    np.save(path + "data_raw_test.npy", X_test_raw)
    np.save(path + "data_no_test.npy", X_test_no)
    np.save(path + "data_ne_test.npy", X_test_ne)
    np.save(path + "data_op_test.npy", X_test_op)
    np.save(path + "data_pe_test.npy", X_test_pe)
    np.save(path + "data_sc_test.npy", X_test_sc)
    np.save(path + "data_ti_test.npy", X_test_ti)


if __name__ == '__main__':
    normal = np.load("/media/zyx/self_supervised/DSADS/normal.npy")
    abnormal = np.load("/media/zyx/self_supervised/DSADS/abnormal.npy")
    number = normal.shape[0]
    dataX = np.vstack((normal,abnormal))

    data_raw, data_no, data_ne, data_op, data_pe, data_sc, data_ti = transformation(dataX)

    data_raw_n = data_raw[:number]
    data_no_n = data_no[:number]
    data_ne_n = data_ne[:number]
    data_op_n = data_op[:number]
    data_pe_n = data_pe[:number]
    data_sc_n = data_sc[:number]
    data_ti_n = data_ti[:number]

    data_raw_a = data_raw[number:]
    data_no_a = data_no[number:]
    data_ne_a = data_ne[number:]
    data_op_a = data_op[number:]
    data_pe_a = data_pe[number:]
    data_sc_a = data_sc[number:]
    data_ti_a = data_ti[number:]

    ####################################save normal data######################
    shuffle( data_raw_n,data_no_n,data_ne_n,data_op_n,data_pe_n,data_sc_n,data_ti_n)

    ####################################save abnormal data######################
    path = '/media/zyx/self_supervised/DSADS/dataset_normalize_together/'
    np.save(path + "data_raw_abnormal.npy", data_raw_a)
    np.save(path + "data_no_abnormal.npy", data_no_a)
    np.save(path + "data_ne_abnormal.npy", data_ne_a)
    np.save(path + "data_op_abnormal.npy", data_op_a)
    np.save(path + "data_pe_abnormal.npy", data_pe_a)
    np.save(path + "data_sc_abnormal.npy", data_sc_a)
    np.save(path + "data_ti_abnormal.npy", data_ti_a)