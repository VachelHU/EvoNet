# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.shapelets import ShapeletModel

class StateRecognition():
    def __init__(self, state_num, inputshape, storepath=''):
        self.state_num = state_num
        self.hislen, self.input_span, self.input_dim = inputshape
        self.modelpath = Path(storepath).absolute()
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)

    def getfeatures(self, x):
        segment_size = x.shape[0]    # [segment size * input span * input dim]

        means = np.reshape(np.mean(x, axis=1), [segment_size, self.input_dim])
        stds = np.reshape(np.std(x, axis=1), [segment_size, self.input_dim])
        features = np.concatenate([means, stds], axis=1)

        # features = np.reshape(x, [segment_size, self.input_span * self.input_dim])

        return features

    def fit(self, x, y=None):
        tempx = np.reshape(x, [-1, self.input_span, self.input_dim])
        tempx = self.getfeatures(tempx)

        self.clu = GaussianMixture(n_components=self.state_num, covariance_type='diag')
        self.clu.fit(tempx)
        joblib.dump(self.clu, self.modelpath / 'states.m')

    def getState(self, x):
        self.clu = joblib.load(self.modelpath / 'states.m')
        tempx = np.reshape(x, [-1, self.input_span, self.input_dim])
        tempx = self.getfeatures(tempx)

        state_proba = self.clu.predict_proba(tempx)
        state_pattern = np.concatenate([self.clu.means_, self.clu.covariances_], axis=-1)

        return np.reshape(state_proba, [-1, self.hislen, self.state_num]).astype(np.float32), np.array(state_pattern, dtype=np.float32)

class SAXRecognition(StateRecognition):
    def __init__(self, state_num, inputshape, storepath=''):
        StateRecognition.__init__(state_num, inputshape, storepath)

    def fit(self, x, y=None):
        sax = SymbolicAggregateApproximation(n_segments=self.hislen, alphabet_size_avg=self.state_num)
        joblib.dump(sax, self.modelpath / 'states.m')

    def getState(self, x):
        sax = joblib.load(self.modelpath / 'states.m')

        sax_dataset_inv = sax.inverse_transform(sax.fit_transform(x))
        uniques = sorted(np.unique(sax_dataset_inv))
        print('sax numbers:', len(uniques))
        state_pattern = np.eye(len(uniques))

        state_proba = np.zeros([x.shape[0], self.hislen, len(uniques)], dtype=np.float)
        tmpstates = np.reshape(sax_dataset_inv, [-1, self.hislen, self.input_span])
        for i in range(tmpstates.shape[0]):
            for j in range(tmpstates.shape[1]):
                index = uniques.index(tmpstates[i, j, 0])
                state_proba[i, j, index] = tmpstates[i, j, 0]

        return np.reshape(state_proba, [-1, self.hislen, self.state_num]).astype(np.float32), np.array(state_pattern, dtype=np.float32)


class ShapeletRecognition(StateRecognition):
    def __init__(self, state_num, inputshape, storepath=''):
        StateRecognition.__init__(state_num, inputshape, storepath)

    def fit(self, x, y=None):
        clf = ShapeletModel(n_shapelets_per_size={self.input_span: self.state_num}, weight_regularizer=.01, verbose_level=0)
        clf.fit(x, y)
        joblib.dump(clf, self.modelpath / 'states.m')

    def getState(self, x):
        clf = joblib.load(self.modelpath / 'states.m')

        shaplets = []
        for s in clf.shapelets_:
            shaplets.append(s)
        shaplets = np.reshape(shaplets, [self.state_num, self.input_span])
        print('shaplets:', shaplets.shape)
        state_pattern = shaplets

        tmpdata = np.reshape(x, [-1, self.hislen, self.input_span])
        state_proba = np.zeros([x.shape[0], self.hislen, self.state_num], dtype=np.float)
        for i in range(x.shape[0]):
            for j in range(self.hislen):
                for k in range(self.state_num):
                    state_proba[i, j, k] = np.sqrt(np.sum(tmpdata[i, j] - shaplets[k]) ** 2)
                state_proba[i, j] = (state_proba[i, j] - min(state_proba[i, j])) / (max(state_proba[i, j]) - min(state_proba[i, j]))
        return np.reshape(state_proba, [-1, self.hislen, self.state_num]).astype(np.float32), np.array(state_pattern,dtype=np.float32)
