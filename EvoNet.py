# -*- coding: utf-8 -*-

import tensorflow as tf

class EvoNet():
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

class EvoLSTM(EvoNet):
    def __init__(self, timesteps, n_nodes, n_features, node_hidden, is_training=True):
        EvoNet.__init__(self, timesteps, n_nodes, n_features, node_hidden, is_training)

    def LSTM_Unit(self, prev_hidden_memory, node_transform):
        with tf.variable_scope('EvoLSTM_Unit'):
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

            # current node hidden
            current_U = tf.reshape(Ut, [-1, self.n_nodes, self.n_features])
            current_node_hidden = tf.reshape(O * tf.nn.tanh(Ut), [-1, self.n_nodes, self.n_features])

        return tf.stack([current_node_hidden, current_U])

    def get_node_hiddens(self, evolution_graph):
        batch_size = tf.shape(evolution_graph)[0]
        # time major
        evolution_graph = tf.transpose(evolution_graph, [1, 0, 2])  # [seq_length * batch_size * n_features]
        evolution_send = evolution_graph[:-1]
        evolution_recieve = evolution_graph[1:]

        initial_node_hidden = tf.ones([batch_size, self.n_nodes, self.n_features], dtype=tf.float32) * self.node_hidden
        initial_cell = tf.stack([initial_node_hidden, initial_node_hidden])

        packed = tf.scan(self.LSTM_Unit, (evolution_send, evolution_recieve), initializer=initial_cell, name="EvoLSTM")

        all_node_hiddens = tf.reshape(packed[:, 0], [self.timesteps, -1, self.n_nodes, self.n_features])
        return all_node_hiddens

class EvoGRU(EvoNet):
    def __init__(self, timesteps, n_nodes, n_features, node_hidden, is_training=True):
        EvoNet.__init__(self, timesteps, n_nodes, n_features, node_hidden, is_training)

    def GRU_Unit(self, prev_hidden_memory, node_transform):
        with tf.variable_scope('EvoGRU_Unit'):
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

            Ut = tf.nn.tanh(tf.matmul(E, self.Wu) + tf.matmul(prev_node_hidden, self.Uu) + self.bu)

            # current node hidden
            current_U = tf.reshape(Ut, [-1, self.n_nodes, self.n_features])
            current_node_hidden = current_U

        return tf.stack([current_node_hidden, current_U])

    def get_node_hiddens(self, evolution_graph):
        batch_size = tf.shape(evolution_graph)[0]
        # time major
        evolution_graph = tf.transpose(evolution_graph, [1, 0, 2])  # [seq_length * batch_size * n_features]
        evolution_send = evolution_graph[:-1]
        evolution_recieve = evolution_graph[1:]

        initial_node_hidden = tf.ones([batch_size, self.n_nodes, self.n_features], dtype=tf.float32) * self.node_hidden
        initial_cell = tf.stack([initial_node_hidden, initial_node_hidden])

        packed = tf.scan(self.GRU_Unit, (evolution_send, evolution_recieve), initializer=initial_cell, name="EvoGRU")

        all_node_hiddens = tf.reshape(packed[:, 0], [self.timesteps, -1, self.n_nodes, self.n_features])
        return all_node_hiddens