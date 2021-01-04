# -*- coding: utf-8 -*-

import os

class ModelParam(object):
    # basic
    model_save_path = "./Repo/model"
    n_jobs = os.cpu_count()

    # dataset
    data_path = './Repo/data'
    data_name = 'netflow'
    his_len = 15
    segment_len = 24
    segment_dim = 2
    n_event = 2
    norm = True

    # state recognition
    n_state = 30
    covariance_type = 'diag'

    # model
    graph_dim = 256
    node_dim = 96
    learning_rate = 0.001
    batch_size = 1000
    id_gpu = '0'
    pos_weight = 1.0
