# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import argparse
import ops
from dataloader import AppLoader
from dataset import getdataset
from EvoNet_tsc import EvoNet_TSC
import loggingOut as logging
import metrics


def fit(model, sess, x, y_clf, trainGeneAssign=False, iteration=100):
    recognition, patterns = model.getState(x, train=trainGeneAssign)

    shuffle_indices, train_indices, validate_indices = ops.splitvalidate(x.shape[0])
    recognition = recognition[shuffle_indices]
    y_clf = y_clf[shuffle_indices]
    x = x[shuffle_indices]

    train_assign = recognition[train_indices]
    train_hidden = patterns
    train_y = y_clf[train_indices]
    train_x = x[train_indices]

    validate_assign = recognition[validate_indices]
    validate_hidden = patterns
    validate_y = y_clf[validate_indices]
    validate_x = x[validate_indices]

    trainloader = AppLoader(model.batch_size)
    trainloader.load_data(train_x, train_y, train_assign, train_hidden)

    validateloader = AppLoader(model.batch_size)
    validateloader.load_data(validate_x, validate_y, validate_assign, validate_hidden)

    bestP = 0.0

    for i in range(iteration):
        loss = model.train(sess, trainloader)
        y_pred, y_true, val_loss, _ = model.use(sess, validateloader)
        metric_result = metrics.predict_accuracy(y_true, y_pred, need_acc=True)
        logstr = 'Epochs {:d}, loss {:f}, vali loss {:f}, Accuracy {:f}'.format(i, loss, val_loss, metric_result['Accuracy'])
        logging.info(logstr)
        if metric_result['Accuracy'] > bestP:
            model.storeModel(sess, model.save_prefix)
            bestP = metric_result['Accuracy']
            print('epoch {} store.'.format(i))


def test(model, sess, x, y_clf):
    recognition, patterns = model.getState(x, train=False)

    testloader = AppLoader(model.batch_size)
    testloader.load_data(x, y_clf, recognition, patterns)

    model.reloadModel(sess, model.save_prefix)
    y_pred, y_true, _, _ = model.use(sess, testloader)
    metric_result = metrics.predict_accuracy(y_true, y_pred, need_acc=True)
    logging.info('Accuracy: {}'.format(metric_result['Accuracy']))

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--statenum", type=int, help="state number")
    parser.add_argument("-d", "--dataset", type=str, choices=['earthquake', 'webtraffic'], help="select the dataset")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("-b", "--batchsize", type=int, default=2000, help="batch size")
    parser.add_argument("-g", "--gpu", type=str, default='0', help="state number")
    parser.add_argument("-p", "--modelpath", type=str, default='./Repo', help="the path of storing model")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainx, trainy, testx, testy = getdataset(dataname=args.dataset)
    trainy = ops.one_hot(trainy, 2)
    testy = ops.one_hot(testy, 2)
    print("train: {}, {} \ntest: {}, {}".format(trainx.shape, trainy.shape, testx.shape, testy.shape))
    args.inputshape = trainx.shape[1:]

    model = EvoNet_TSC(args)

    print('training.')
    tf.reset_default_graph()
    model.buildNets(is_training=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        fit(model, sess, trainx, trainy, trainGeneAssign=True, iteration=100)

    print("testing.")
    model.buildNets(is_training=False)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        test(model, sess, testx, testy)
        sess.close()
