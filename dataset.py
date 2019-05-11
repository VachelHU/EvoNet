# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def getPublicDataset(name):
    traindata = pd.read_csv('./data/public_dataset/UCRArchive_2018/{}/{}_TRAIN.tsv'.format(name, name), sep='\t', header=None)
    testdata = pd.read_csv('./data/public_dataset/UCRArchive_2018/{}/{}_TEST.tsv'.format(name, name), sep='\t', header=None)
    return traindata, testdata


def getdataset(datashape, rate=0.8, dataname='ims'):

    if dataname == 'earthquake':
        traindata, testdata = getPublicDataset('Earthquakes')

        trainY = traindata[0].values
        trainX = traindata.values[:, 9:]
        trainX = np.reshape(trainX, [-1, 21, 24, 1])

        testY = testdata[0].values
        testX = testdata.values[:, 9:]
        testX = np.reshape(testX, [-1, 21, 24, 1])

        return trainX, trainY, testX, testY

    elif dataname == 'worm':
        traindata, testdata = getPublicDataset('WormsTwoClass')

        trainY = traindata[0].values - 1
        trainX = traindata.values[:, 1:]
        trainX = np.reshape(trainX, [-1, 15, 60, 1])

        testY = testdata[0].values - 1
        testX = testdata.values[:, 1:]
        testX = np.reshape(testX, [-1, 15, 60, 1])

        return trainX, trainY, testX, testY

    elif dataname == 'stock':
        dataset = np.load('./data/public_dataset/djia30stock/djia30stock.npy')
        his_length = 50
        x = []
        y = []

        dataset = np.nan_to_num(dataset, 0)
        for i in range(his_length, dataset.shape[1], 5):
            tempx = dataset[:, i - his_length:i]
            tempx_ = dataset[:, i, :, -1]
            tempy = (np.std(tempx_, axis=1) > 1.0).astype(np.int32)

            x.append(tempx)
            y.append(tempy)

        x = np.reshape(np.concatenate(x, axis=0), [-1, his_length, 5, 4])
        y = np.concatenate(y, axis=0)

        shuffle_index = np.random.permutation(np.arange(x.shape[0]))
        x = x[shuffle_index]
        y = y[shuffle_index]

        trainX = x[:int(x.shape[0] * 0.8)]
        trainY = y[:int(x.shape[0] * 0.8)]
        testX = x[int(x.shape[0] * 0.8):]
        testY = y[int(x.shape[0] * 0.8):]

        return trainX, trainY, testX, testY

    elif dataname == 'googletraffic':
        x = np.load('./data/public_dataset/googletraffic/datax.npy')
        x = np.log1p(np.abs(x))
        x = np.reshape(x, [-1, 12, 30, 1])
        y = np.load('./data/public_dataset/googletraffic/datay.npy')

        shuffle_index = np.random.permutation(np.arange(x.shape[0]))
        x = x[shuffle_index]
        y = y[shuffle_index]

        trainX = x[:int(x.shape[0] * 0.8)]
        trainY = y[:int(x.shape[0] * 0.8)]
        testX = x[int(x.shape[0] * 0.8):]
        testY = y[int(x.shape[0] * 0.8):]

        return trainX, trainY, testX, testY

    else:
        return -1