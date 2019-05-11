# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

class GMM():
    def __init__(self, gene_num, inputshape=(12, 4, 2), modelname='', storepath=''):
        self.gene_num = gene_num
        self.hislen, self.input_span, self.input_dim = inputshape
        self.algorithmName = modelname
        self.modelpath = Path(storepath).absolute()
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)

    def getfeatures(self, x):
        sample_size = x.shape[0]

        means = np.reshape(np.mean(x, axis=1), [sample_size, self.input_dim])
        stds = np.reshape(np.std(x, axis=1), [sample_size, self.input_dim])
        # polys = np.polyfit(np.arange(self.input_span, dtype=np.int32),
        #                    np.reshape(np.transpose(x, [1, 0, 2]), [self.input_span, sample_size * self.input_dim]),
        #                    3)
        # polys = np.transpose(np.reshape(polys, [4, sample_size, self.input_dim]), [1, 0, 2])
        # features = np.concatenate([means, stds, np.reshape(polys, [sample_size, 4 * self.input_dim])], axis=1)

        features = np.concatenate([means, stds], axis=1)
        # features = np.reshape(x, [sample_size, self.input_span * self.input_dim])

        return features

    def fit(self, x):
        tempx = np.reshape(x, [-1, self.input_span, self.input_dim])
        tempx = self.getfeatures(tempx)

        self.clu = GaussianMixture(n_components=self.gene_num, covariance_type='diag')
        self.clu.fit(tempx)
        joblib.dump(self.clu, self.modelpath / (self.algorithmName+'.m'))

    def getGene(self, x):
        self.clu = joblib.load(self.modelpath / (self.algorithmName+'.m'))
        tempx = np.reshape(x, [-1, self.input_span, self.input_dim])
        tempx = self.getfeatures(tempx)

        gene_assign = self.clu.predict_proba(tempx)
        gene_latent = np.concatenate([self.clu.means_, self.clu.covariances_], axis=-1)

        return np.reshape(gene_assign, [-1, self.hislen, self.gene_num]).astype(np.float32), np.array(gene_latent, dtype=np.float32)


class geneAssignment():
    def __init__(self, gene_num, inputshape=(12, 4, 2), gene_rnn_units=16, latent_dim=16, lr=0.001, batch_size=3000,
                 storepath='', modelname='', confidence=0.001, trainCepochs=20, trainGepoch=20):
        self.gene_num = gene_num
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.gene_rnn_units = gene_rnn_units
        self.lr = lr
        self.inputshape = inputshape
        self.confidence = confidence
        self.trainCepochs = trainCepochs
        self.trainGepochs = trainGepoch
        self.modelname = modelname
        self.modelpath = Path(storepath).absolute()
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)

    def selectAlgorithm(self, algorithm):
        self.model = GMM(self.gene_num, self.inputshape, self.modelname, self.modelpath)

    def trainmodel(self, x):
        self.model.fit(x)

    def usemodel(self, x):
        return self.model.getGene(x)
