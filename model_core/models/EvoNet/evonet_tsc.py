# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.externals import joblib

from model_core.models.EvoNet.evonet import EvoNet
from model_core.templates.interface import BaseMLModelTemplate


class EvoNet_TSC(BaseMLModelTemplate):

    def build_model(self, is_training):

        self.x = tf.placeholder(tf.float32, [None, self.param.his_len, self.param.segment_len, self.param.segment_dim], name='input_raw_sequence')
        self.a = tf.placeholder(tf.float32, [None, self.param.his_len, self.param.n_state], name='input_state_sequence')
        self.p = tf.placeholder(tf.float32, [self.param.n_state, self.param.node_dim], name='input_state_features')
        self.y = tf.placeholder(tf.int32, [None, self.param.his_len+1], name='input_event_label')

        y_clf = tf.one_hot(self.y, self.param.n_event)

        model = EvoNet()
        model.set_configuration(self.param, is_training=is_training)
        graph_logits, node_logits, attention_logits = model.get_embedding(self.a, y_clf, self.p)

        self.attention_score = tf.nn.softmax(attention_logits)

        # output
        patterns = tf.reshape(graph_logits, [-1, self.param.graph_dim])
        net = tf.layers.dense(patterns, 512, activation=tf.nn.relu, name="outnet_fc1")
        out_logits = tf.layers.dense(net, self.param.n_event, activation=tf.nn.relu, name='outnet_fc2')

        # loss and train
        self.net_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.reshape(y_clf[:, 2:], [-1, self.param.n_event]), out_logits))
        self.net_optim = tf.train.AdamOptimizer(self.param.learning_rate, beta1=0.7).minimize(self.net_loss)

        # classification test
        patterns_g = tf.reduce_mean(graph_logits, axis=1)
        patterns_n = tf.reduce_mean(tf.reshape(node_logits, [-1, (self.param.his_len-1)*self.param.n_state, self.param.node_dim]), axis=1)
        patterns_x = tf.reduce_mean(tf.reshape(self.x, [-1, self.param.his_len, self.param.segment_len*self.param.segment_dim]), axis=1)
        self.patterns = tf.concat([patterns_g, patterns_n, patterns_x], axis=-1)

        self.clf = XGBClassifier(n_estimators=500, max_depth=6, n_jobs=self.param.n_jobs, scale_pos_weight=self.param.pos_weight)

        self.saver = tf.train.Saver()

        return 0

    def fit(self, sess, dataloader=None):

        if not dataloader:
            raise Exception('no data input')

        loss = []
        global_features = []
        for _ in range(dataloader.num_batch):
            x_batch, y_batch, prob_batch, patterns = dataloader.next_batch()
            feeds = {
                self.a: prob_batch,
                self.p: patterns,
                self.y: y_batch,
                self.x: x_batch
            }
            _, loss_net, feature_ = sess.run([self.net_optim, self.net_loss, self.patterns], feed_dict=feeds)
            loss.append(loss_net)
            global_features.append(feature_)

        global_features = np.concatenate(global_features)
        self.clf.fit(global_features, dataloader.y[:, -1])

        return np.mean(loss)

    def predict(self, sess, dataloader=None):

        if not dataloader:
            raise Exception('no data input')

        global_features = []
        for _ in range(dataloader.num_batch):
            x_batch, y_batch, prob_batch, patterns = dataloader.next_batch()
            feeds = {
                self.a: prob_batch,
                self.p: patterns,
                self.y: y_batch,
                self.x: x_batch
            }
            feature_ = sess.run(self.patterns, feed_dict=feeds)
            global_features.append(feature_)
        global_features = np.concatenate(global_features)
        return self.clf.predict(global_features), self.clf.predict_proba(global_features)

    def getAttention(self, sess, dataloader=None):

        if not dataloader:
            raise Exception('no data input')

        global_attentions = []
        for _ in range(dataloader.num_batch):
            x_batch, y_batch, prob_batch, patterns = dataloader.next_batch()
            feeds = {
                self.a: prob_batch,
                self.p: patterns,
                self.y: y_batch,
                self.x: x_batch
            }
            a_ = sess.run(self.attention_score, feed_dict=feeds)
            global_attentions.append(a_)

        return np.concatenate(global_attentions)


    def store(self, path, sess=None):
        save_model_name = "ETNet_{}_{}".format(self.param.data_name, self.param.n_state)
        if not os.path.exists(os.path.join(path, save_model_name)):
            os.makedirs(os.path.join(path, save_model_name))

        self.saver.save(sess, os.path.join(path, save_model_name, 'model'))
        joblib.dump(self.clf, os.path.join(path, save_model_name, "clf.model"))

    def restore(self, path, sess=None):
        save_model_name = "ETNet_{}_{}".format(self.param.data_name, self.param.n_state)
        self.saver.restore(sess, os.path.join(path, save_model_name, 'model'))
        self.clf = joblib.load(os.path.join(path, save_model_name, "clf.model"))