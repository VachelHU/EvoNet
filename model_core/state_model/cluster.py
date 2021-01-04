# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from model_core.templates.interface import BaseMLModelTemplate


class ClusterStateRecognition(BaseMLModelTemplate):

    def build_model(self):
        self.model_obj = GaussianMixture(n_components=self.param.n_state, covariance_type=self.param.covariance_type)

    def fit(self, ts, **kwargs):
        _, _, segment_len, segment_dim = ts.shape

        ts_ = np.reshape(ts, [-1, segment_len, segment_dim])
        ts_ = self.__getfeatures__(ts_)
        try:
            self.model_obj.fit(ts_)
            self.store(self.param.model_save_path)
        except Exception as e:
            raise e

    def predict(self, x):
        ts = np.reshape(x, [-1, x.shape[-2], x.shape[-1]])
        ts = self.__getfeatures__(ts)
        self.restore(self.param.model_save_path)
        tprob = self.model_obj.predict_proba(ts)
        tpatterns = np.concatenate([self.model_obj.means_, self.model_obj.covariances_], axis=1)

        xprob = np.reshape(tprob, [-1, x.shape[1], self.param.n_state])

        return xprob, tpatterns

    # get cluster features
    def __getfeatures__(self, x):
        segment_len = x.shape[1]    # [segment size * segment len * metric dim]
        segment_dim = x.shape[2]

        # means = np.reshape(np.mean(x, axis=1), [-1, segment_dim])
        # stds = np.reshape(np.std(x, axis=1), [-1, segment_dim])
        # features = np.concatenate([means, stds], axis=1)

        features = np.reshape(x, [-1, segment_len * segment_dim])

        return features

    def store(self, path, **kwargs):
        save_model_name = "gmm_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        joblib.dump(self.model_obj, os.path.join(path, save_model_name))

    def restore(self, path, **kwargs):
        save_model_name = "gmm_{}_{}.state_model".format(self.param.data_name, self.param.n_state)
        self.model_obj = joblib.load(os.path.join(path, save_model_name))