# -*- coding: utf-8 -*-

import os
import pickle
import tensorflow as tf
import argparse
from model_core.config import ModelParam
from data_factory import LoadData, BatchLoader

from model_core.state_model import ClusterStateRecognition
from model_core.models import EvoNet_TSC
import model_core.metrics as mt

datainfos = {'djia30': [50, 5, 4, 3.0],
             'webtraffic': [12, 30, 1, 3.0],
             'netflow': [15, 24, 2, 6.0],
             'clockerr': [12, 4, 2, 6.0]}


def main(dataname, gpu=0):
    params = ModelParam()
    params.data_name = dataname
    params.his_len = datainfos[params.data_name][0]
    params.segment_len = datainfos[params.data_name][1]
    params.segment_dim = datainfos[params.data_name][2]
    params.node_dim = 2 * params.segment_dim * params.segment_len
    params.id_gpu = '{}'.format(gpu)
    params.pos_weight = datainfos[params.data_name][3]
    params.learning_rate = 0.001

    os.environ["CUDA_VISIBLE_DEVICES"] = params.id_gpu

    dataloader = LoadData()
    dataloader.set_configuration(params)

    trainx, trainy, testx, testy = dataloader.fetch_data()
    rawx, rawy = dataloader.fetch_raw_data()
    print(rawx.shape, rawy.shape, trainx.shape, trainy.shape, testx.shape, testy.shape)
    n = trainx.shape[0]+testx.shape[0]
    y1 = sum(trainy[:,-1])+sum(testy[:, -1])
    print(n, y1/n, rawx.shape[0] * rawx.shape[1])


    # state
    print("state recognizing...")
    state_model = ClusterStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model()
    # state_model.fit(rawx)
    train_prob, train_patterns = state_model.predict(trainx)
    test_prob, test_patterns = state_model.predict(testx)
    print(train_patterns.shape, train_prob.shape, test_patterns.shape, test_prob.shape)

    # establish dataloader
    trainloader = BatchLoader(params.batch_size)
    trainloader.load_data(trainx, trainy, train_prob, train_patterns, shuffle=True)
    testloader = BatchLoader(params.batch_size)
    testloader.load_data(testx, testy, test_prob, test_patterns, shuffle=False)

    # model
    model = EvoNet_TSC()

    model.set_configuration(params)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    print('model training...')
    with tf.Session(config=config) as sess:
        model.build_model(is_training=True)
        init_vars = tf.global_variables_initializer()
        sess.run(init_vars)

        bestP = 0.0
        for i in range(100):
            loss = model.fit(sess, trainloader)
            y_pred, y_pred_prob = model.predict(sess, testloader)
            results = mt.predict_accuracy(testy[:, -1], y_pred)
            auc = mt.predict_auc(testy[:, -1], y_pred_prob[:, 1])
            logstr = 'Epochs {:d}, loss {:f}, Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}, AUC {:f}'.format(i, loss, results['Accuracy'], results['Precision'], results['Recall'], results['F1'], auc)
            print(logstr)
            p = 2 * results['Precision'] * results['Recall'] / (results['Precision'] + results['Recall'])
            if p > bestP:
                model.store(params.model_save_path, sess=sess)
                bestP = p
                print('epoch {} store.'.format(i))


    print("model testing...")
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.build_model(is_training=False)
        model.restore(params.model_save_path, sess=sess)
        y_pred, y_pred_prob = model.predict(sess, testloader)
        results = mt.predict_accuracy(testy[:, -1], y_pred)
        auc = mt.predict_auc(testy[:, -1], y_pred_prob[:, 1])
        logstr = 'Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}, AUC {:f}'.format(results['Accuracy'], results['Precision'], results['Recall'], results['F1'], auc)
        print(logstr)


def getattention(dataname, gpu=0):
    params = ModelParam()
    params.data_name = dataname
    params.his_len = datainfos[params.data_name][0]
    params.segment_len = datainfos[params.data_name][1]
    params.segment_dim = datainfos[params.data_name][2]
    params.node_dim = 2 * params.segment_dim * params.segment_len
    params.id_gpu = '{}'.format(gpu)
    params.pos_weight = datainfos[params.data_name][3]
    params.learning_rate = 0.001

    os.environ["CUDA_VISIBLE_DEVICES"] = params.id_gpu

    dataloader = LoadData()
    dataloader.set_configuration(params)

    trainx, trainy, _, _ = dataloader.fetch_data()
    rawx = trainx
    rawy = trainy
    print(rawx.shape, rawy.shape)

    # state
    print("state recognizing...")
    state_model = ClusterStateRecognition()
    state_model.set_configuration(params)
    state_model.build_model()
    prob, patterns = state_model.predict(rawx)
    print(patterns.shape, prob.shape)

    # establish dataloader
    testloader = BatchLoader(params.batch_size)
    testloader.load_data(rawx, rawy, prob, patterns, shuffle=False)

    # model
    model = EvoNet_TSC()
    model.set_configuration(params)
    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.build_model(is_training=False)
        model.restore(params.model_save_path, sess=sess)

        y_pred, y_pred_prob = model.predict(sess, testloader)
        attentions = model.getAttention(sess, testloader)

        results = mt.predict_accuracy(rawy[:, -1], y_pred)
        auc = mt.predict_auc(rawy[:, -1], y_pred_prob[:, 1])
        logstr = 'Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}, AUC {:f}'.format(results['Accuracy'], results['Precision'], results['Recall'], results['F1'], auc)
        print(logstr)

        store_obj = {'x': rawx, 'y': rawy, 'prob': prob, 'pattern':patterns, 'attention':attentions}
        pickle.dump(store_obj, open('./Repo/output/result_{}.pkl'.format(dataname), 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=['djia30', 'webtraffic', 'netflow', 'clockerr'], default='djia30', help="select dataset")
    parser.add_argument("-g", "--gpu", type=str, choices=['0', '1', '2'], default='0', help="target gpu id")
    args = parser.parse_args()

    main(args.dataset, gpu=args.gpu)
    getattention(args.dataset, gpu=args.gpu)