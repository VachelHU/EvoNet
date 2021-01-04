# -*- coding: utf-8 -*-
# @Time : 2019-11-02 10:43 
# @Author : VachelHU
# @File : saxword.py 
# @Software: PyCharm

import os
import numpy as np
from sklearn.externals import joblib
from model_core.templates.interface import BaseMLModelTemplate

from tslearn.piecewise import SymbolicAggregateApproximation


class SAXStateRecognition(BaseMLModelTemplate):

    def build_model(self, **kwargs):
        self.his_len = kwargs['his_len']
        self.segment_dim = kwargs['segment_dim']
        self.model_obj = SymbolicAggregateApproximation(n_segments=self.his_len, alphabet_size_avg=self.param.n_state)

    def fit(self, x, y=None):
        self.store(self.param.model_save_path)

    def predict(self, x):
        self.restore(self.param.model_save_path)

        sax_dataset_inv = self.model_obj.inverse_transform(self.model_obj.fit_transform(x))
        uniques = sorted(np.unique(sax_dataset_inv))
        print('sax numbers:', len(uniques))
        state_pattern = np.eye(len(uniques))

        state_proba = np.zeros([x.shape[0], self.his_len, len(uniques)], dtype=np.float)
        tmpstates = np.reshape(sax_dataset_inv, [-1, self.his_len, self.segment_dim])
        for i in range(tmpstates.shape[0]):
            for j in range(tmpstates.shape[1]):
                index = uniques.index(tmpstates[i, j, 0])
                state_proba[i, j, index] = tmpstates[i, j, 0]

        return np.reshape(state_proba, [-1, self.his_len, self.param.n_state]).astype(np.float32), np.array(state_pattern, dtype=np.float32)

    def store(self, path, **kwargs):
        save_model_name = "sax_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        joblib.dump(self.model_obj, os.path.join(path, save_model_name))

    def restore(self, path, **kwargs):
        save_model_name = "sax_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        self.model_obj = joblib.load(os.path.join(path, save_model_name))