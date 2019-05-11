# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pathlib import Path
import sys, os
import ops
from dataloader import AppLoader
from dataset import getdataset
from geneAssignment import geneAssignment
from xgboost import XGBClassifier
import loggingOut as logging
import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class EGLSTM():
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std), regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])

    def __init__(self, timesteps, n_nodes, n_features, node_hidden, is_training=True):
        self.timesteps = timesteps - 1
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.node_hidden = node_hidden

        if is_training:
            self.Win = self.init_weights(self.n_features, self.n_features, name='Recieve_hidden_weight')
            self.bin = self.init_bias(self.n_features, name='Recieve_hidden_bias')

            self.Wout = self.init_weights(self.n_features, self.n_features, name='Send_hidden_weight')
            self.bout = self.init_bias(self.n_features, name='Send_hidden_bias')

            self.Wi = self.init_weights(2 * self.n_features, self.n_features, name='Input_Hidden_weight')
            self.Ui = self.init_weights(self.n_features, self.n_features, name='Input_State_weight')
            self.bi = self.init_bias(self.n_features, name='Input_Hidden_bias')

            self.Wf = self.init_weights(2 * self.n_features, self.n_features, name='Forget_Hidden_weight')
            self.Uf = self.init_weights(self.n_features, self.n_features, name='Forget_State_weight')
            self.bf = self.init_bias(self.n_features, name='Forget_Hidden_bias')

            self.Wo = self.init_weights(2 * self.n_features, self.n_features, name='Output_Hidden_weight')
            self.Uo = self.init_weights(self.n_features, self.n_features, name='Output_State_weight')
            self.bo = self.init_bias(self.n_features, name='Output_Hidden_bias')

            self.Wu = self.init_weights(2 * self.n_features, self.n_features, name='Global_Hidden_weight')
            self.Uu = self.init_weights(self.n_features, self.n_features, name='Global_State_weight')
            self.bu = self.init_bias(self.n_features, name='Global_Hidden_bias')
        else:
            self.Win = self.no_init_weights(self.n_features, self.n_features, name='Recieve_hidden_weight')
            self.bin = self.no_init_bias(self.n_features, name='Recieve_hidden_bias')

            self.Wout = self.no_init_weights(self.n_features, self.n_features, name='Send_hidden_weight')
            self.bout = self.no_init_bias(self.n_features, name='Send_hidden_bias')

            self.Wi = self.no_init_weights(2 * self.n_features, self.n_features, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.n_features, self.n_features, name='Input_State_weight')
            self.bi = self.no_init_bias(self.n_features, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(2 * self.n_features, self.n_features, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.n_features, self.n_features, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.n_features, name='Forget_Hidden_bias')

            self.Wo = self.no_init_weights(2 * self.n_features, self.n_features, name='Output_Hidden_weight')
            self.Uo = self.no_init_weights(self.n_features, self.n_features, name='Output_State_weight')
            self.bo = self.no_init_bias(self.n_features, name='Output_Hidden_bias')

            self.Wu = self.no_init_weights(2 * self.n_features, self.n_features, name='Global_Hidden_weight')
            self.Uu = self.no_init_weights(self.n_features, self.n_features, name='Global_State_weight')
            self.bu = self.no_init_bias(self.n_features, name='Global_Hidden_bias')

    def EGLSTM_Unit(self, prev_hidden_memory, node_transform):
        with tf.variable_scope('EGLSTM_Unit'):
            prev_node_hidden, prev_U = tf.unstack(prev_hidden_memory)

            send_nodes = node_transform[0]
            receive_nodes = node_transform[1]

            Min = tf.matmul(tf.reshape(send_nodes, [-1, self.n_nodes, 1]), tf.reshape(receive_nodes, [-1, 1, self.n_nodes]))
            Mout = tf.matmul(tf.reshape(receive_nodes, [-1, self.n_nodes, 1]), tf.reshape(send_nodes, [-1, 1, self.n_nodes]))

            # propagation
            Ein = tf.reshape(tf.matmul(Min, prev_node_hidden), [-1, self.n_features])
            Ein = tf.nn.tanh(tf.matmul(Ein, self.Win) + self.bin)

            Eout = tf.reshape(tf.matmul(Mout, prev_node_hidden), [-1, self.n_features])
            Eout = tf.nn.tanh(tf.matmul(Eout, self.Wout) + self.bout)

            E = tf.concat([Ein, Eout], axis=-1)

            prev_node_hidden = tf.reshape(prev_node_hidden, [-1, self.n_features])
            prev_U = tf.reshape(prev_U, [-1, self.n_features])

            #input gate
            I = tf.nn.sigmoid(tf.matmul(E, self.Wi) + tf.matmul(prev_node_hidden, self.Ui) + self.bf)
            # foreget gate
            F = tf.nn.sigmoid(tf.matmul(E, self.Wf) + tf.matmul(prev_node_hidden, self.Uf) + self.bf)
            # output gate
            O = tf.nn.sigmoid(tf.matmul(E, self.Wo) + tf.matmul(prev_node_hidden, self.Uo) + self.bo)

            # global attributes
            U_ = tf.nn.tanh(tf.matmul(E, self.Wu) + tf.matmul(F * prev_node_hidden, self.Uu) + self.bu)

            # output
            Ut = F * prev_U + I * U_

            # Ut = tf.nn.tanh(tf.matmul(E, self.Wu) + tf.matmul(prev_node_hidden, self.Uu) + self.bu)

            # current node hidden
            current_U = tf.reshape(Ut, [-1, self.n_nodes, self.n_features])
            current_node_hidden = tf.reshape(O * tf.nn.tanh(Ut), [-1, self.n_nodes, self.n_features])
            # current_node_hidden = current_U

        return tf.stack([current_node_hidden, current_U])

    def get_node_hiddens(self, evolution_graph):
        batch_size = tf.shape(evolution_graph)[0]
        # time major
        evolution_graph = tf.transpose(evolution_graph, [1, 0, 2])  # [seq_length * batch_size * n_features]
        evolution_send = evolution_graph[:-1]
        evolution_recieve = evolution_graph[1:]

        initial_node_hidden = tf.ones([batch_size, self.n_nodes, self.n_features], dtype=tf.float32) * self.node_hidden
        initial_cell = tf.stack([initial_node_hidden, initial_node_hidden])

        packed = tf.scan(self.EGLSTM_Unit, (evolution_send, evolution_recieve), initializer=initial_cell, name="EGLSTM")

        all_node_hiddens = tf.reshape(packed[:, 0], [self.timesteps, -1, self.n_nodes, self.n_features])
        return all_node_hiddens


class SEG():
    def __init__(self, args):
        self.params = args
        self.gene_num = args['model']['overall']['gene_num']
        self.latent_dim = args['model']['overall']['latent_dim']
        self.event_num = args['model']['geneapplication']['event_num']
        self.hislen, self.input_span, self.input_dim = args['model']['overall']['inputshape']
        self.batch_size = args['model']['overall']['batch_size']
        self.lr = args['model']['overall']['learning_rate']

        self.beta = 0.7
        self.modelpath = Path(args['save']['repo']).absolute()
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
        self.save_prefix = self.params['save']['dataset'] + '_' + self.params['save']['appmodel'] + "_fc"

    def getGene(self, x, train=False):
        self.assignmodel = geneAssignment(gene_num=self.params['model']['overall']['gene_num'],
                    inputshape=self.params['model']['overall']['inputshape'],
                    latent_dim=self.params['model']['overall']['latent_dim'],
                    lr=self.params['model']['overall']['learning_rate'],
                    batch_size=self.params['model']['overall']['batch_size'],
                    gene_rnn_units=self.params['model']['geneassignment']['gene_rnn_units'],
                    storepath=self.params['save']['repo'],
                    modelname=self.params['save']['dataset'] + '_' + self.params['save']['assignmodel'],
                    confidence=self.params['model']['geneassignment']['confidence'],
                    trainCepochs=self.params['model']['geneassignment']['trainCepoch'],
                    trainGepoch=self.params['model']['geneassignment']['trainGepoch'])

        self.assignmodel.selectAlgorithm(algorithm=self.params['save']['assignmodel'])
        if train:
            self.assignmodel.trainmodel(x)
        geneassign, genelatent = self.assignmodel.usemodel(x)
        return geneassign, genelatent

    def buildNets(self, is_training):
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, [None, self.hislen, self.input_span, self.input_dim])
        self.va = tf.placeholder(tf.float32, [None, self.hislen, self.gene_num], name='input_vertex_assign')
        self.vh = tf.placeholder(tf.float32, [self.gene_num, self.latent_dim], name='input_vertex_hidden')
        self.y_clf = tf.placeholder(tf.float32, [None, self.event_num], name='input_event_label')

        lstm = EGLSTM(self.hislen, self.gene_num, self.latent_dim, self.vh, is_training=is_training)
        self.gene_hiddens = lstm.get_node_hiddens(self.va)

        net = tf.reshape(self.gene_hiddens[-1], [-1, self.gene_num * self.latent_dim])
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
        # y_pred = np.concatenate(y_pred, axis=0)
        y_pred = self.clf.predict(net)
        y_true = np.argmax(dataloader.y_clf_labels, axis=1)
        hidden = np.concatenate(hidden, axis=0)
        return y_pred, y_true, np.mean(loss), hidden


    def fit(self, sess, trainx, trainy, testx, testy, trainGeneAssign=False, iteration=100):
        # assign, hidden = self.getGene(x, train=trainGeneAssign)

        # shuffle_indices, train_indices, validate_indices = ops.splitvalidate(x.shape[0])
        # assign = assign[shuffle_indices]
        # y_clf = y_clf[shuffle_indices]
        # x = x[shuffle_indices]
        #
        # train_assign = assign[train_indices]
        # train_hidden = hidden
        # train_y_clf = y_clf[train_indices]
        # train_x = x[train_indices]

        train_assign, train_hidden = self.getGene(trainx, train=trainGeneAssign)
        trainloader = AppLoader(self.batch_size)
        trainloader.load_data(trainx, trainy, train_assign, train_hidden)

        # validate_assign = assign[validate_indices]
        # validate_hidden = hidden
        # validate_y_clf = y_clf[validate_indices]
        # validate_x = x[validate_indices]

        test_assign, test_hidden = self.getGene(testx, train=False)
        validateloader = AppLoader(self.batch_size)
        validateloader.load_data(testx, testy, test_assign, test_hidden)

        bestF1 = 0.0

        for i in range(iteration):
            loss = self.train(sess, trainloader)
            y_pred, y_true, val_loss, _ = self.use(sess, validateloader)
            metric_result = metrics.predict_accuracy(y_true, y_pred, need_acc=False)
            logstr = 'Epochs {:d}, loss {:f}, vali loss {:f}, Precision {:f}, Recall {:f}, F1 {:f}'.format(i, loss, val_loss, metric_result['Precision'], metric_result['Recall'], metric_result['F1'])
            # logstr = 'Epochs {:d}, loss {:f}, vali loss {:f}, Accuracy {:f}'.format(i, loss, val_loss, metric_result['Accuracy'])
            logging.info(logstr)
            # if metric_result['F1'] > bestF1:
            #     self.storeModel(sess, self.save_prefix)
            #     bestF1 = metric_result['F1']
            #     print('epoch {} store.'.format(i))

    def test(self, sess, x, y_clf):
        assign, hidden = self.getGene(x)

        testloader = AppLoader(self.batch_size)
        testloader.load_data(x, y_clf, assign, hidden)

        self.reloadModel(sess, self.save_prefix)
        y_pred, y_true, _, _ = self.use(sess, testloader)
        metric_result = metrics.predict_accuracy(y_true, y_pred)
        print('Precision {:f}, Recall {:f}, F1 {:f}'.format(metric_result['Precision'], metric_result['Recall'], metric_result['F1']))

    def storeModel(self, sess, save_prefix):
        self.saver.save(sess, str(self.modelpath / save_prefix))

    def reloadModel(self, sess, save_prefix):
        self.saver.restore(sess, str(self.modelpath / save_prefix))

if __name__ == '__main__':
    args = ops.parse_args('./config.json')
    trainx, trainy, testx, testy = getdataset(datashape=args['model']['overall']['inputshape'], dataname=args['save']['dataset'])
    trainy = ops.one_hot(trainy, args['model']['geneapplication']['event_num'])
    testy = ops.one_hot(testy, args['model']['geneapplication']['event_num'])
    print("train: {}, {} \ntest: {}, {}".format(trainx.shape, trainy.shape, testx.shape, testy.shape))
    model = SEG(args)

    print('training.')
    tf.reset_default_graph()
    model.buildNets(is_training=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
    #     writer = tf.summary.FileWriter("./Repo/TensorBoard/eglstm", sess.graph)
    # writer.close()
        model.fit(sess, trainx, trainy, testx, testy, trainGeneAssign=True, iteration=100)

    # print("testing.")
    # model.buildNets(is_training=False)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     model.test(sess, testx, testy)
    #     sess.close()
