# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABCMeta, abstractmethod


class BaseMLModelTemplate(object, metaclass=ABCMeta):

    def __init__(self):
        self.ts = None
        self.labels = None
        self.model_obj = None
        self.param = None

    def set_configuration(self, model_param):
        self.param = model_param
        if not os.path.exists(self.param.model_save_path):
            os.makedirs(self.param.model_save_path)

    @abstractmethod
    def build_model(self, **kwargs):
        """
        User Defined Model
        :param kwargs:
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, ts, **kwargs):
        """
        User Defined Model
        :param input data
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        :param: input data
        :return: the predicted labels and the predicted probability
        """
        pass

    @abstractmethod
    def store(self, path, **kwargs):
        """
        :param: store path
        :return: None
        """
        pass

    @abstractmethod
    def restore(self, path, **kwargs):
        """
        :param: store path
        :return: None
        """
        pass

    @property
    def name(self):
        """Get Model Name"""
        return self.__class__.__name__
