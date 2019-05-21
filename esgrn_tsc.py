# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pathlib import Path
from state_recognition import StateRecognition
from ESGRN import ESGLSTM
from xgboost import XGBClassifier
from sklearn.externals import joblib

class ESGRN_TSC():
    def __init__(self, args):
        self.args = args
        self.state_num = args.statenum
        self.event_num = 2
        self.hislen, self.input_span, self.input_dim = args.inputshape
        self.pattern_dim = self.input_dim * 4

        self.batch_size = args.batchsize
        self.lr = args.learning_rate
        self.beta = 0.7
        self.modelpath = Path(args.modelpath).absolute()
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)

        self.save_prefix = args.dataset + "_fc"

    def getState(self, x, train=False):
        self.statemodel = StateRecognition(self.state_num, self.args.inputshape, self.args.modelpath)

        if train:
            self.statemodel.fit(x)
        state_proba, state_pattern = self.statemodel.getState(x)
        return state_proba, state_pattern

    def buildNets(self, is_training):
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, [None, self.hislen, self.input_span, self.input_dim])
        self.va = tf.placeholder(tf.float32, [None, self.hislen, self.state_num], name='input_vertex_assign')
        self.vh = tf.placeholder(tf.float32, [self.state_num, self.pattern_dim], name='input_vertex_hidden')
        self.y_clf = tf.placeholder(tf.float32, [None, self.event_num], name='input_event_label')

        lstm = ESGLSTM(self.hislen, self.state_num, self.pattern_dim, self.vh, is_training=is_training)
        self.gene_hiddens = lstm.get_node_hiddens(self.va)

        net = tf.reshape(self.gene_hiddens[-1], [-1, self.state_num * self.pattern_dim])
        net = tf.concat([net, tf.reshape(self.x, [-1, self.hislen*self.input_span*self.input_dim])], axis=-1)

        self.net = net

        self.clf = XGBClassifier(n_jobs=40)

        outnet = tf.layers.dense(net, 512, activation=tf.nn.relu, name="outnet_fc1")
        out_logits = tf.layers.dense(outnet, self.event_num, activation=tf.nn.relu, name='outnet_fc2')
        pred = tf.nn.softmax(out_logits, name="output_event_pred")

        # loss
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y_clf, logits=out_logits))

        # optimizers
        self.optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta).minimize(self.loss)

        self.y_pred = tf.argmax(pred, axis=1, name='output_pred')

        self.saver = tf.train.Saver()

        return 0

    def train(self, sess, dataloader):
        loss = []
        net = []
        dataloader.reset_pointer()
        for _ in range(dataloader.num_batch):
            x_batch, y_clf_batch, assign_batch, hidden_batch = dataloader.next_batch()
            feeds = {
                self.va: assign_batch,
                self.vh: hidden_batch,
                self.y_clf: y_clf_batch,
                self.x: x_batch
            }
            _, loss_, net_ = sess.run([self.optim, self.loss, self.net], feed_dict=feeds)
            loss.append(loss_)
            net.append(net_)
        net = np.concatenate(net, axis=0)
        self.clf.fit(net, np.argmax(dataloader.y_clf_labels, axis=1))
        return np.mean(loss)

    def use(self, sess, dataloader):
        y_pred = []
        loss = []
        net = []
        hidden = []
        dataloader.reset_pointer()
        for _ in range(dataloader.num_batch):
            x_batch, y_clf_batch, assign_batch, hidden_batch = dataloader.next_batch()
            feeds = {
                self.va: assign_batch,
                self.vh: hidden_batch,
                self.y_clf: y_clf_batch,
                self.x: x_batch
            }
            temppred, loss_, net_, h_ = sess.run([self.y_pred, self.loss, self.net, self.gene_hiddens], feed_dict=feeds)
            y_pred.append(temppred)
            loss.append(loss_)
            net.append(net_)
            hidden.append(np.transpose(h_, [1, 0, 2, 3]))
        net = np.concatenate(net, axis=0)
        y_pred = self.clf.predict(net)
        y_true = np.argmax(dataloader.y_clf_labels, axis=1)
        hidden = np.concatenate(hidden, axis=0)
        return y_pred, y_true, np.mean(loss), hidden

    def storeModel(self, sess, save_prefix):
        self.saver.save(sess, str(self.modelpath / save_prefix))
        joblib.dump(self.clf, self.modelpath / 'tsc.m')

    def reloadModel(self, sess, save_prefix):
        self.saver.restore(sess, str(self.modelpath / save_prefix))
        self.clf = joblib.load(self.modelpath / 'tsc.m')