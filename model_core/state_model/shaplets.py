# -*- coding: utf-8 -*-
# @Time : 2019-11-02 10:43 
# @Author : VachelHU
# @File : shaplets.py 
# @Software: PyCharm


import os
import numpy as np
from sklearn.externals import joblib
from model_core.templates.interface import BaseMLModelTemplate
from tslearn.shapelets import ShapeletModel


class ShapletStateRecognition(BaseMLModelTemplate):

    def build_model(self, **kwargs):
        self.his_len = kwargs['his_len']
        self.segment_dim = kwargs['segment_dim']

        self.model_obj = ShapeletModel(n_shapelets_per_size={self.segment_dim: self.param.n_state}, weight_regularizer=.01, verbose_level=0)

    def fit(self, x, y=None):
        self.model_obj.fit(x, y)
        self.store(self.param.model_save_path)

    def predict(self, x):
        self.restore(self.param.model_save_path)

        shaplets = []
        for s in self.model_obj.shapelets_:
            shaplets.append(s)
        shaplets = np.reshape(shaplets, [self.param.n_state, self.segment_dim])
        print('shaplets:', shaplets.shape)
        state_pattern = shaplets

        tmpdata = np.reshape(x, [-1, self.his_len, self.segment_dim])
        state_proba = np.zeros([x.shape[0], self.his_len, self.param.n_state], dtype=np.float)
        for i in range(x.shape[0]):
            for j in range(self.his_len):
                for k in range(self.param.n_state):
                    state_proba[i, j, k] = np.sqrt(np.sum(tmpdata[i, j] - shaplets[k]) ** 2)
                state_proba[i, j] = (state_proba[i, j] - min(state_proba[i, j])) / (max(state_proba[i, j]) - min(state_proba[i, j]))
        return np.reshape(state_proba, [-1, self.his_len, self.param.n_state]).astype(np.float32), np.array(state_pattern,dtype=np.float32)

    def store(self, path, **kwargs):
        save_model_name = "shaplet_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        joblib.dump(self.model_obj, os.path.join(path, save_model_name))

    def restore(self, path, **kwargs):
        save_model_name = "shaplet_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        self.model_obj = joblib.load(os.path.join(path, save_model_name))