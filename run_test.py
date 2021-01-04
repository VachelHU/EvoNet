# -*- coding: utf-8 -*-
# @Time : 2020-10-01 23:32
# @Author : VachelHU
# @File : run_test.py.py 
# @Software: PyCharm

from model_core.config import ModelParam
from data_factory.takedata import LoadData
from model_core.state_model import ClusterStateRecognition, SAXStateRecognition, ShapletStateRecognition


def main():
    params = ModelParam()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.id_gpu

    dataloader = LoadData()
    dataloader.set_configuration(params)

    trainx, trainy, testx, testy = dataloader.fetch_data()
    rawx, rawy = dataloader.fetch_raw_data()
    print(rawx.shape, rawy.shape, trainx.shape, trainy.shape, testx.shape, testy.shape)

    state_model = ClusterStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model()
    state_model.fit(rawx)
    train_patterns, train_prob = state_model.predict(trainx)
    print(train_patterns.shape, train_prob.shape)

    state_model = SAXStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model(his_len=15, segment_dim=2)
    state_model.fit(rawx)
    train_patterns, train_prob = state_model.predict(trainx)
    print(train_patterns.shape, train_prob.shape)

    state_model = ShapletStateRecognition
    state_model.set_configuration(params)
    state_model.build_model(his_len=15, segment_dim=2)
    state_model.fit(rawx, rawy)
    train_patterns, train_prob = state_model.predict(trainx)
    print(train_patterns.shape, train_prob.shape)



if __name__ == '__main__':
    main()