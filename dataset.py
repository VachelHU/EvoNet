# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def getdataset(dataname='ims'):

    if dataname == 'earthquake':
        traindata = pd.read_csv('./data/public_dataset/Earthquakes/Earthquakes_TRAIN.tsv', sep='\t', header=None)
        testdata = pd.read_csv('./data/public_dataset/Earthquakes/Earthquakes_TEST.tsv', sep='\t', header=None)

        trainY = traindata[0].values
        trainX = traindata.values[:, 9:]
        trainX = np.reshape(trainX, [-1, 21, 24, 1])

        testY = testdata[0].values
        testX = testdata.values[:, 9:]
        testX = np.reshape(testX, [-1, 21, 24, 1])

        return trainX, trainY, testX, testY

    elif dataname == 'webtraffic':
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