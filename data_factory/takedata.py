# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


class LoadData(object):
    """
    Fetch raw samples from different sources.
    """
    def __init__(self):
        self.param = None
        self.data_preprocess = None
        self.raw_data_array = {'data': None, 'label': None}
        self.data_array = {'data': None, 'label': None}
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

    def set_configuration(self, config):
        """Init Data configuration"""
        self.param = config


    def __get_local_data__(self):
        """Fetch raw local data"""
        if self.param.data_name in ['djia30', 'webtraffic', 'netflow', 'clockerr']:
            self.raw_data_array['data'] = np.load(os.path.join(self.param.data_path, self.param.data_name, 'series.npy'))
            self.raw_data_array['label'] = np.load(os.path.join(self.param.data_path, self.param.data_name, 'events.npy'))

        else:
            raise Exception('no data')

    def __split_on_time__(self, start, end):
        tempx = []
        tempy = []
        for i in range(start, end - 1):
            tempx.append(self.x[:, i - self.param.his_len:i])
            tempy.append(self.y[:, i - self.param.his_len:i + 1])
        tempx = np.concatenate(tempx, axis=0)
        tempy = np.concatenate(tempy, axis=0)
        return tempx, tempy

    def __clean_data_by_entities__(self):
        """process the raw data"""

        if self.param.data_name == 'djia30':
            x = self.raw_data_array['data']
            if self.param.norm:
                x = self.__normalize__(x)
            y = self.raw_data_array['label']

            self.x = np.reshape(x, [-1, 518, self.param.segment_len, self.param.segment_dim])
            self.y = np.reshape(y, [-1, 518])

            # divide it to train and test
            self.x_train, self.y_train = self.__split_on_time__(self.param.his_len, int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8))
            self.x_test, self.y_test = self.__split_on_time__(int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8), self.x.shape[1])


        elif self.param.data_name == 'webtraffic':
            x = np.log1p(np.abs(self.raw_data_array['data']))
            if self.param.norm:
                x = self.__normalize__(x)
            y = self.raw_data_array['label']

            self.x = np.reshape(x, [-1, 26, self.param.segment_len, self.param.segment_dim])
            self.y = np.reshape(y, [-1, 26])

            print(self.x.shape, self.y.shape)

            # reduce data capacity
            indexes = np.random.choice(self.x.shape[0], size=21234, replace=False)
            self.x = self.x[indexes]
            self.y = self.y[indexes]

            # divide it to train and test
            self.x_train, self.y_train = self.__split_on_time__(self.param.his_len, int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8))
            self.x_test, self.y_test = self.__split_on_time__(int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8), self.x.shape[1])


        elif self.param.data_name == 'netflow':
            x = np.log1p(np.abs(self.raw_data_array['data']))
            if self.param.norm:
                x = self.__normalize__(x)
            y = self.raw_data_array['label']
            y[y > 0] = 1

            self.x = np.reshape(x, [-1, 40, self.param.segment_len, self.param.segment_dim])
            self.y = np.reshape(y, [-1, 40])

            # divide it to train and test
            self.x_train, self.y_train = self.__split_on_time__(self.param.his_len, int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8))
            self.x_test, self.y_test = self.__split_on_time__(int(self.param.his_len + (self.x.shape[1] - self.param.his_len) * 0.8), self.x.shape[1])


        elif self.param.data_name == 'clockerr':
            x = np.log1p(np.abs(self.raw_data_array['data']))
            if self.param.norm:
                x = self.__normalize__(x)
            y = self.raw_data_array['label']
            y[y > 0] = 1

            self.x = np.reshape(x, [-1, 12, self.param.segment_len, self.param.segment_dim])
            self.y = np.reshape(y, [-1, 13])

            print(self.x.shape, self.y.shape)

            # reduce data capacity
            indexes = np.random.choice(self.x.shape[0], size=151234, replace=False)
            self.x = self.x[indexes]
            self.y = self.y[indexes]


            sf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
            trainindex, testindex = list(sf.split(self.x, self.y[:, -1]))[0]

            # divide it to train and test
            self.x_train, self.y_train = self.x[trainindex], self.y[trainindex]
            self.x_test, self.y_test = self.x[testindex], self.y[testindex]

        else:
            raise Exception('no data')


    def __normalize__(self, data):
        shape = data.shape
        tempshape = [shape[0], shape[1] * shape[2], shape[3]]
        data_ = np.reshape(data, tempshape)
        data_ = (data_ - np.mean(data_, axis=1)[:, np.newaxis]) / (np.std(data_, axis=1)[:, np.newaxis] + 1e-20)
        return np.reshape(data_, shape)


    def fetch_data(self):
        print("load {} data".format(self.param.data_name))
        self.__get_local_data__()
        self.__clean_data_by_entities__()

        return self.x_train, self.y_train, self.x_test, self.y_test

    def fetch_raw_data(self):
        return self.x, self.y


if __name__ == '__main__':
    pass